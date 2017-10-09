#pragma SDS data zero_copy(ip)
#pragma SDS data zero_copy(op)
#pragma SDS data access_pattern(ip:SEQUENTIAL)
#pragma SDS data mem_attribute(ip:PHYSICAL_CONTIGUOUS)
#pragma SDS data mem_attribute(op:PHYSICAL_CONTIGUOUS)
void CQT_conv2d_1_3x3_hw(FP16 ip[173056], FP16 op[173056], FP16 weight[9], int bias, int act, int last);

#pragma SDS data zero_copy(ip)
#pragma SDS data zero_copy(op)
#pragma SDS data access_pattern(ip:SEQUENTIAL)
#pragma SDS data mem_attribute(ip:PHYSICAL_CONTIGUOUS)
#pragma SDS data mem_attribute(op:PHYSICAL_CONTIGUOUS)
void CQT_conv2d_2_3x3_hw(FP16 ip[43264], FP16 op[43264], FP16 weight[9], int bias, int act, int last);

#pragma SDS data zero_copy(ip)
#pragma SDS data zero_copy(op)
#pragma SDS data access_pattern(ip:SEQUENTIAL)
#pragma SDS data mem_attribute(ip:PHYSICAL_CONTIGUOUS)
#pragma SDS data mem_attribute(op:PHYSICAL_CONTIGUOUS)
void CQT_conv2d_3_3x3_hw(FP16 ip[10816], FP16 op[10816], FP16 weight[9], int bias, int act, int last);

#pragma SDS data zero_copy(ip)
#pragma SDS data zero_copy(op)
#pragma SDS data access_pattern(ip:SEQUENTIAL)
#pragma SDS data mem_attribute(ip:PHYSICAL_CONTIGUOUS)
#pragma SDS data mem_attribute(op:PHYSICAL_CONTIGUOUS)
void CQT_conv2d_4_3x3_hw(FP16 ip[2704], FP16 op[2704], FP16 weight[9], int bias, int act, int last);

#pragma SDS data zero_copy(ip)
#pragma SDS data zero_copy(op)
#pragma SDS data access_pattern(ip:SEQUENTIAL)
#pragma SDS data mem_attribute(ip:PHYSICAL_CONTIGUOUS)
#pragma SDS data mem_attribute(op:PHYSICAL_CONTIGUOUS)
void CQT_conv2d_5_3x3_hw(FP16 ip[676], FP16 op[676], FP16 weight[9], int bias, int act, int last);

#pragma SDS data zero_copy(ip)
#pragma SDS data zero_copy(op)
#pragma SDS data access_pattern(ip:SEQUENTIAL)
#pragma SDS data mem_attribute(ip:PHYSICAL_CONTIGUOUS)
#pragma SDS data mem_attribute(op:PHYSICAL_CONTIGUOUS)
void CQT_conv2d_6_3x3_hw(FP16 ip[169], FP16 op[169], FP16 weight[9], int bias, int act, int last);

#pragma SDS data zero_copy(ip)
#pragma SDS data zero_copy(op)
#pragma SDS data access_pattern(ip:SEQUENTIAL)
#pragma SDS data mem_attribute(ip:PHYSICAL_CONTIGUOUS)
#pragma SDS data mem_attribute(op:PHYSICAL_CONTIGUOUS)
void CQT_conv2d_7_3x3_hw(FP16 ip[169], FP16 op[169], FP16 weight[9], int bias, int act, int last);

#pragma SDS data zero_copy(ip)
#pragma SDS data zero_copy(op)
#pragma SDS data access_pattern(ip:SEQUENTIAL)
#pragma SDS data mem_attribute(ip:PHYSICAL_CONTIGUOUS)
#pragma SDS data mem_attribute(op:PHYSICAL_CONTIGUOUS)
void CQT_conv2d_8_3x3_hw(FP16 ip[169], FP16 op[169], FP16 weight[9], int bias, int act, int last);

