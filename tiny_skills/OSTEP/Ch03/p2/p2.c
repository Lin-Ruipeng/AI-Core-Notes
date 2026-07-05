#include <stdio.h>
#include <stdlib.h>
#include <sys/wait.h>
#include <unistd.h>

int main(int argc, char *argv[]) {
  printf("hello world (pid: %d)\n", (int)getpid());
  int rc = fork();

  if (rc < 0) {
    fprintf(stderr, "fork failed\n");
    exit(1);
  } else if (rc == 0) {
    printf("Hello, I am child (pid: %d)\n", (int)getpid());
  } else {
    int wc = wait(NULL);  // 这里是等待, 等待NULL也就是子进程运行结束 !
                          // 因为在父进程看来, 子进程的PID为0 !
    printf("Hello, I am parent of %d (wc: %d) (pid: %d)\n", rc, wc, (int)getpid());
  }

  return 0;
}
