#!/bin/bash

SRC_DIR=../hero-sim-proto
DST_DIR=./
protoc -I=$SRC_DIR --python_out=$DST_DIR $SRC_DIR/result.proto