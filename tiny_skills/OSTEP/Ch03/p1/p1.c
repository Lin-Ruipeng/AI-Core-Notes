#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int main(int argc, char *argv[]) {
  // 打印当前进程PID
  printf("Hello world (pid: %d)\n", (int)getpid());
  int rc = fork();

  if (rc < 0) {
    fprintf(stderr, "fork failed\n");
    exit(1);
  } else if (rc == 0) {
    // fork()返回值为0表示执行成功,成功创建了子进程(复制自己)
    printf("Hello, I am child (pid: %d)\n", (int)getpid());
  } else {
    // fork()成功会返回子进程的PID以便于管理
    printf("Hello, I am parent of %d (pid: %d)\n", rc, (int)getpid());
  }

  return 0;
}
