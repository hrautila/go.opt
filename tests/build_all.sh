#!/bin/bash

NAMES="conelp coneqp cp cpl gp lp qp socp sdp mcsdp qcl1"

for N in $NAMES
do
    echo "building " test${N}
    go build test${N}.go
done


