#!/bin/bash

cmake -B build

cmake --build build

./build/p4

# 默认不保存文件
if [ -z "$1" ]; then
    echo "为您自动打开输出文件: p4.output , 其文件内容为:"
    cat p4.output
    echo "以上是文件内容, 已为您自动清理输出文件, 如果您需要保留文件, 请您运行:"
    echo "$0 save"
    rm "./p4.output"
    exit 0
fi

# 传入保存参数
if [ "$1" == "save" ]; then
    echo "为您自动打开输出文件: p4.output , 其文件内容为:"
    cat p4.output
    echo "以上是文件内容"
    exit 0
else 
    echo "错误: 未知参数 '$1' "
    echo "用法: $0 save"
    exit 1
fi

