# `p1` 程序说明

## 快速开始

一键运行

```bash
bash ./p1.sh
```

## 运行结果

```bash
Hello world (pid: 10988)
Hello, I am parent of 10989 (pid: 10988)
Hello, I am child (pid: 10989)
```

注：PID值每次运行均不一样

## 结果说明

本程序主要是使用了Linux API的 `fork()` 接口，
使用此函数会将自身程序复制一份并运行，且注意到是连寄存器状态都复制了，
所以会发现第一行的Hello world只打印了一次。

接着注意到，fork()之后会有两个进程运行——父进程和子进程。
其中，子进程在fork()处获得的返回值为0，表示fork成功。
父进程在fork()处获取到的返回值为子进程的PID。

## 其他

本程序源码来自于书籍《OSTEP》
