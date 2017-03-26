//
// Created by natu on 17/03/22.
//

#include <stdio.h>

#include "inc/cqt.h"
#include "inc/cqt_net.h"
#include "cqt_gen/cqt_gen.h"


int main(void)
{
    CQT_NET *vgg16_p;

    vgg16_p = cqt_init();
    printf("hello cqt\n");

    return 0;
}
