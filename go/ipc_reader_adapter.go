// Copyright (c) 2025 ADBC Drivers Contributors
//
// This file has been modified from its original version, which is
// under the Apache License:
//
// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

package databricks

import (
	"bytes"
	"context"
	"database/sql/driver"
	"errors"
	"fmt"
	"io"
	"strings"
	"sync/atomic"

	"github.com/apache/arrow-adbc/go/adbc"
	"github.com/apache/arrow-go/v18/arrow"
	"github.com/apache/arrow-go/v18/arrow/array"
	"github.com/apache/arrow-go/v18/arrow/ipc"
	dbsqlrows "github.com/databricks/databricks-sql-go/rows"
)

// ipcReaderAdapter uses the new IPC stream interface for Arrow access
type ipcReaderAdapter struct {
	rows          driver.Rows
	ipcIterator   dbsqlrows.ArrowIPCStreamIterator
	currentReader *ipc.Reader
	currentRecord arrow.RecordBatch
	schema        *arrow.Schema
	closed        bool
	refCount      int64
	err           error
}

// newIPCReaderAdapter creates a RecordReader using direct IPC stream access
func newIPCReaderAdapter(ctx context.Context, rows driver.Rows) (array.RecordReader, error) {
	ipcRows, ok := rows.(dbsqlrows.Rows)
	if !ok {
		return nil, adbc.Error{
			Code: adbc.StatusInternal,
			Msg:  "[db] rows do not support Arrow IPC streams",
		}
	}

	// Get IPC stream iterator
	ipcIterator, err := ipcRows.GetArrowIPCStreams(ctx)
	if err != nil {
		return nil, adbc.Error{
			Code: adbc.StatusInternal,
			Msg:  fmt.Sprintf("failed to get IPC streams: %v", err),
		}
	}

	adapter := &ipcReaderAdapter{
		rows:        rows,
		refCount:    1,
		ipcIterator: ipcIterator,
	}

	// Load the first IPC stream to get the schema.
	// Note: SchemaBytes() may return empty bytes if no direct results were
	// returned with the query response. The schema is populated lazily
	// during the first data fetch in databricks-sql-go. By loading the
	// first reader, we ensure the schema is available.
	err = adapter.loadNextReader()
	if err != nil && err != io.EOF {
		return nil, adbc.Error{
			Code: adbc.StatusInternal,
			Msg:  fmt.Sprintf("failed to initialize IPC reader: %v", err),
		}
	}

	// Get schema from the first reader, or fall back to SchemaBytes() if
	// the result set is empty (no readers available)
	if adapter.currentReader != nil {
		adapter.schema = adapter.currentReader.Schema()
	} else {
		// Empty result set - try to get schema from SchemaBytes()
		schema_bytes, err := ipcIterator.SchemaBytes()
		if err != nil {
			return nil, adbc.Error{
				Code: adbc.StatusInternal,
				Msg:  fmt.Sprintf("failed to get schema bytes: %v", err),
			}
		}

		if len(schema_bytes) == 0 {
			// Workaround for https://github.com/databricks/databricks-sql-go/pull/327
			// SchemaBytes() can be empty when databricks-sql-go doesn't
			// propagate the schema for 0-row results. Fall back to
			// building the schema from driver.Rows column metadata.
			schema, err := schemaFromRowsMetadata(rows)
			if err != nil {
				return nil, adbc.Error{
					Code: adbc.StatusInternal,
					Msg:  fmt.Sprintf("schema bytes are empty and failed to build schema from column metadata: %v", err),
				}
			}
			adapter.schema = schema
			return adapter, nil
		}

		reader, err := ipc.NewReader(bytes.NewReader(schema_bytes))
		if err != nil {
			return nil, adbc.Error{
				Code: adbc.StatusInternal,
				Msg:  fmt.Sprintf("failed to read schema: %v", err),
			}
		}
		adapter.schema = reader.Schema()
		reader.Release()
	}

	// The server non-deterministically omits Spark:DataType:SqlName
	// metadata from the IPC schema. Ensure all fields have metadata
	// using driver.Rows column type info (always available via thrift).
	adapter.schema = ensureSchemaMetadata(adapter.schema, rows)

	if adapter.schema == nil {
		return nil, adbc.Error{
			Code: adbc.StatusInternal,
			Msg:  "schema is nil",
		}
	}

	return adapter, nil
}

// schemaFromRowsMetadata builds an Arrow schema from driver.Rows column
// metadata. This is used as a fallback when SchemaBytes() is empty for
// 0-row result sets: https://github.com/databricks/databricks-sql-go/pull/327
func schemaFromRowsMetadata(rows driver.Rows) (*arrow.Schema, error) {
	typed, ok := rows.(driver.RowsColumnTypeDatabaseTypeName)
	if !ok {
		return nil, fmt.Errorf("driver.Rows does not implement RowsColumnTypeDatabaseTypeName")
	}

	nullableTyped, hasNullable := rows.(driver.RowsColumnTypeNullable)

	columns := rows.Columns()
	fields := make([]arrow.Field, len(columns))
	for i, name := range columns {
		dbType := typed.ColumnTypeDatabaseTypeName(i)
		nullable := true
		if hasNullable {
			if n, ok := nullableTyped.ColumnTypeNullable(i); ok {
				nullable = n
			}
		}
		fields[i] = arrow.Field{
			Name:     name,
			Type:     databricksTypeToArrow(dbType),
			Nullable: nullable,
			// Spark:DataType:SqlName metadata for consistency with the
			// server-provided schema and the databricks-sql-go logic.
			Metadata: arrow.MetadataFrom(map[string]string{
				"Spark:DataType:SqlName": sparkSqlNameFromDBType(dbType),
			}),
		}
	}
	return arrow.NewSchema(fields, nil), nil
}

// databricksTypeToArrow maps a Databricks SQL type name to an Arrow data type.
func databricksTypeToArrow(dbType string) arrow.DataType {
	switch strings.ToUpper(dbType) {
	case "BOOLEAN":
		return arrow.FixedWidthTypes.Boolean
	case "BYTE", "TINYINT":
		return arrow.PrimitiveTypes.Int8
	case "SHORT", "SMALLINT":
		return arrow.PrimitiveTypes.Int16
	case "INT", "INTEGER":
		return arrow.PrimitiveTypes.Int32
	case "LONG", "BIGINT":
		return arrow.PrimitiveTypes.Int64
	case "FLOAT":
		return arrow.PrimitiveTypes.Float32
	case "DOUBLE":
		return arrow.PrimitiveTypes.Float64
	case "STRING":
		return arrow.BinaryTypes.String
	case "BINARY":
		return arrow.BinaryTypes.Binary
	case "DATE":
		return arrow.FixedWidthTypes.Date32
	case "TIMESTAMP", "TIMESTAMP_NTZ":
		return arrow.FixedWidthTypes.Timestamp_us
	case "DECIMAL":
		return &arrow.Decimal128Type{Precision: 38, Scale: 18}
	default:
		return arrow.BinaryTypes.String
	}
}

// sparkSqlNameFromDBType converts a database type name to a Spark SQL type name
// suitable for Spark:DataType:SqlName metadata. For DECIMAL, the precision and
// scale (38,18) are a default placeholder because ColumnTypeDatabaseTypeName
// returns just "DECIMAL" with no way to determine the actual precision or scale.
// Consumers that convert Utf8 to Decimal infer the actual precision and scale
// from the data values at query time.
func sparkSqlNameFromDBType(dbType string) string {
	upper := strings.ToUpper(dbType)
	if upper == "DECIMAL" {
		return "DECIMAL(38,18)"
	}
	return upper
}

// ensureSchemaMetadata adds Spark:DataType:SqlName metadata to schema fields
// that are missing it. The Databricks server non-deterministically omits
// metadata from the Arrow IPC schema. We use driver.Rows column type info
// (always available via thrift) to fill in the gaps.
func ensureSchemaMetadata(schema *arrow.Schema, rows driver.Rows) *arrow.Schema {
	typed, ok := rows.(driver.RowsColumnTypeDatabaseTypeName)
	if !ok {
		return schema
	}

	fields := schema.Fields()
	changed := false
	newFields := make([]arrow.Field, len(fields))

	for i, f := range fields {
		newFields[i] = f
		if _, ok := f.Metadata.GetValue("Spark:DataType:SqlName"); ok {
			continue
		}
		if dbType := typed.ColumnTypeDatabaseTypeName(i); dbType != "" {
			newFields[i].Metadata = arrow.MetadataFrom(map[string]string{
				"Spark:DataType:SqlName": sparkSqlNameFromDBType(dbType),
			})
			changed = true
		}
	}

	if !changed {
		return schema
	}
	md := schema.Metadata()
	return arrow.NewSchema(newFields, &md)
}

func (r *ipcReaderAdapter) loadNextReader() error {
	if r.currentReader != nil {
		r.currentReader.Release()
		r.currentReader = nil
	}

	// Get next IPC stream
	if !r.ipcIterator.HasNext() {
		return io.EOF
	}

	ipcStream, err := r.ipcIterator.Next()
	if err != nil {
		return err
	}

	// Create IPC reader from stream
	reader, err := ipc.NewReader(ipcStream)
	if err != nil {
		return adbc.Error{
			Code: adbc.StatusInternal,
			Msg:  fmt.Sprintf("failed to create IPC reader: %v", err),
		}
	}

	r.currentReader = reader

	return nil
}

// Implement array.RecordReader interface
func (r *ipcReaderAdapter) Schema() *arrow.Schema {
	return r.schema
}

func (r *ipcReaderAdapter) Next() bool {
	if r.closed || r.err != nil {
		return false
	}

	// Release previous record
	if r.currentRecord != nil {
		r.currentRecord.Release()
		r.currentRecord = nil
	}

	// Try to get next record from current reader
	if r.currentReader != nil && r.currentReader.Next() {
		r.currentRecord = r.currentReader.RecordBatch()
		r.currentRecord.Retain()
		return true
	}

	// Need to load next IPC stream
	err := r.loadNextReader()
	if err == io.EOF {
		return false
	} else if err != nil {
		r.err = err
		return false
	}

	// Try again with new reader
	if r.currentReader != nil && r.currentReader.Next() {
		r.currentRecord = r.currentReader.RecordBatch()
		r.currentRecord.Retain()
		return true
	}

	return false
}

func (r *ipcReaderAdapter) Record() arrow.RecordBatch {
	return r.currentRecord
}

func (r *ipcReaderAdapter) RecordBatch() arrow.RecordBatch {
	return r.currentRecord
}

func (r *ipcReaderAdapter) Release() {
	if atomic.AddInt64(&r.refCount, -1) <= 0 {
		if r.closed {
			panic("Double cleanup on ipc_reader_adapter - was Release() called with a closed reader?")
		}
		r.closed = true

		if r.currentRecord != nil {
			r.currentRecord.Release()
			r.currentRecord = nil
		}

		if r.currentReader != nil {
			r.currentReader.Release()
			r.currentReader = nil
		}

		if r.schema != nil {
			r.schema = nil
		}

		r.ipcIterator.Close()

		if r.rows != nil {
			r.err = errors.Join(r.err, r.rows.Close())
			r.rows = nil
		}
	}
}

func (r *ipcReaderAdapter) Retain() {
	atomic.AddInt64(&r.refCount, 1)
}

func (r *ipcReaderAdapter) Err() error {
	return r.err
}
