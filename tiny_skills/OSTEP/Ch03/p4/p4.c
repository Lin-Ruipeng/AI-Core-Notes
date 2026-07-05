#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

int main(int argc, char* argv[]) {
  int rc = fork();

  if (rc < 0) {
    fprintf(stderr, "fork failed!\n");
    exit(1);
  } else if (rc == 0) {
    // 输出重定向!
    close(STDOUT_FILENO);  // 关闭标准输出! 不输出到屏幕!
    open("./p4.output", O_CREAT | O_WRONLY | O_TRUNC, S_IRWXU);
    // 把标准输出重定向到这个文件, 所以原本屏幕上的内容被输出到文件里了

    // 等价代码: wc p4.c > ./p4.output

    // 执行子程序
    char* myargs[3];
    myargs[0] = strdup("wc");
    myargs[1] = strdup("p4.c");
    myargs[2] = NULL;
    execvp(myargs[0], myargs);

  } else {
    int wc = wait(NULL);
  }

  return 0;
}
