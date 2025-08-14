#ifndef __UTIL_H__
#define __UTIL_H__

void print_MNIST_digit(unsigned char image_array[28][28]) 
{
  //row,col -> image y,x (the intuitive way)
  for (unsigned i = 0; i < 28; i++) {
    for (unsigned j = 0; j < 28; j++) {
      if (image_array[i][j]>30)
        printf("# ");
      else
        printf("  ");
    }
    printf("\n");
  }
  printf("\n");
}

#endif
