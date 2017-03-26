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
    int ret;

    vgg16_p = cqt_init();
    printf("hello cqt\n");

    ret = cqt_load_weight_from_files(vgg16_p, "weight/");
    if (ret != CQT_RET_OK) {
        printf("ERROR in cqt_load_weight_from_files %d\n", ret);
    }

    return 0;
}
