#include "theory.h"
// #include "SMTLIB/Core.h"
// #include "SMTLIB/BufferRef.h"
// #include "SMTLIB/Float.h"
// #include "SMTLIB/Messages.h"
// #include <stdint.h>
// #include <stdlib.h>
__device__ int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  if (size != 28) {

    return 0;
  }
  BufferRef<const uint8_t> jfs_buffer_ref =
      BufferRef<const uint8_t>(data, size);
  const Float<8, 24> x0 = makeFloatFrom<8, 24>(jfs_buffer_ref, 0, 31);
  const Float<8, 24> x1 = makeFloatFrom<8, 24>(jfs_buffer_ref, 32, 63);
  const Float<8, 24> x2 = makeFloatFrom<8, 24>(jfs_buffer_ref, 64, 95);
  const Float<8, 24> x3 = makeFloatFrom<8, 24>(jfs_buffer_ref, 96, 127);
  const Float<8, 24> x4 = makeFloatFrom<8, 24>(jfs_buffer_ref, 128, 159);
  const Float<8, 24> x5 = makeFloatFrom<8, 24>(jfs_buffer_ref, 160, 191);
  const Float<8, 24> x6 = makeFloatFrom<8, 24>(jfs_buffer_ref, 192, 223);
  const BitVector<1> jfs_ssa_0 = BitVector<1>(UINT64_C(1));
  const BitVector<8> jfs_ssa_1 = BitVector<8>(UINT64_C(130));
  const BitVector<23> jfs_ssa_2 = BitVector<23>(UINT64_C(2097152));
  const Float<8, 24> jfs_ssa_3 = Float<8, 24>(jfs_ssa_0, jfs_ssa_1, jfs_ssa_2);
  const bool jfs_ssa_4 = jfs_ssa_3.fpleq(x0);
  const BitVector<1> jfs_ssa_5 = BitVector<1>(UINT64_C(0));
  const Float<8, 24> jfs_ssa_6 = Float<8, 24>(jfs_ssa_5, jfs_ssa_1, jfs_ssa_2);
  const bool jfs_ssa_7 = x0.fpleq(jfs_ssa_6);
  const bool jfs_ssa_8 = jfs_ssa_4 && jfs_ssa_7;
  if (jfs_ssa_8) {
  } else {
    return 0;
  }
  const bool jfs_ssa_9 = jfs_ssa_3.fpleq(x1);
  const bool jfs_ssa_10 = x1.fpleq(jfs_ssa_6);
  const bool jfs_ssa_11 = jfs_ssa_9 && jfs_ssa_10;
  if (jfs_ssa_11) {
  } else {
    return 0;
  }
  const bool jfs_ssa_12 = jfs_ssa_3.fpleq(x2);
  const bool jfs_ssa_13 = x2.fpleq(jfs_ssa_6);
  const bool jfs_ssa_14 = jfs_ssa_12 && jfs_ssa_13;
  if (jfs_ssa_14) {
  } else {
    return 0;
  }
  const bool jfs_ssa_15 = jfs_ssa_3.fpleq(x3);
  const bool jfs_ssa_16 = x3.fpleq(jfs_ssa_6);
  const bool jfs_ssa_17 = jfs_ssa_15 && jfs_ssa_16;
  if (jfs_ssa_17) {
  } else {
    return 0;
  }
  const bool jfs_ssa_18 = jfs_ssa_3.fpleq(x4);
  const bool jfs_ssa_19 = x4.fpleq(jfs_ssa_6);
  const bool jfs_ssa_20 = jfs_ssa_18 && jfs_ssa_19;
  if (jfs_ssa_20) {
  } else {
    return 0;
  }
  const bool jfs_ssa_21 = jfs_ssa_3.fpleq(x5);
  const bool jfs_ssa_22 = x5.fpleq(jfs_ssa_6);
  const bool jfs_ssa_23 = jfs_ssa_21 && jfs_ssa_22;
  if (jfs_ssa_23) {
  } else {
    return 0;
  }
  const bool jfs_ssa_24 = jfs_ssa_3.fpleq(x6);
  const bool jfs_ssa_25 = x6.fpleq(jfs_ssa_6);
  const bool jfs_ssa_26 = jfs_ssa_24 && jfs_ssa_25;
  if (jfs_ssa_26) {
  } else {
    return 0;
  }
  const BitVector<8> jfs_ssa_27 = BitVector<8>(UINT64_C(0));
  const BitVector<23> jfs_ssa_28 = BitVector<23>(UINT64_C(0));
  const Float<8, 24> jfs_ssa_29 =
      Float<8, 24>(jfs_ssa_5, jfs_ssa_27, jfs_ssa_28);
  const BitVector<8> jfs_ssa_30 = BitVector<8>(UINT64_C(126));
  const BitVector<23> jfs_ssa_31 = BitVector<23>(UINT64_C(5872025));
  const Float<8, 24> jfs_ssa_32 =
      Float<8, 24>(jfs_ssa_5, jfs_ssa_30, jfs_ssa_31);
  const Float<8, 24> jfs_ssa_33 = x0.mul(JFS_RM_RNE, jfs_ssa_32);
  const Float<8, 24> jfs_ssa_34 = jfs_ssa_29.add(JFS_RM_RNE, jfs_ssa_33);
  const BitVector<8> jfs_ssa_35 = BitVector<8>(UINT64_C(125));
  const BitVector<23> jfs_ssa_36 = BitVector<23>(UINT64_C(6375341));
  const Float<8, 24> jfs_ssa_37 =
      Float<8, 24>(jfs_ssa_0, jfs_ssa_35, jfs_ssa_36);
  const Float<8, 24> jfs_ssa_38 = x1.mul(JFS_RM_RNE, jfs_ssa_37);
  const Float<8, 24> jfs_ssa_39 = jfs_ssa_34.add(JFS_RM_RNE, jfs_ssa_38);
  const BitVector<23> jfs_ssa_40 = BitVector<23>(UINT64_C(50332));
  const Float<8, 24> jfs_ssa_41 =
      Float<8, 24>(jfs_ssa_0, jfs_ssa_30, jfs_ssa_40);
  const Float<8, 24> jfs_ssa_42 = x2.mul(JFS_RM_RNE, jfs_ssa_41);
  const Float<8, 24> jfs_ssa_43 = jfs_ssa_39.add(JFS_RM_RNE, jfs_ssa_42);
  const BitVector<8> jfs_ssa_44 = BitVector<8>(UINT64_C(123));
  const BitVector<23> jfs_ssa_45 = BitVector<23>(UINT64_C(5167382));
  const Float<8, 24> jfs_ssa_46 =
      Float<8, 24>(jfs_ssa_0, jfs_ssa_44, jfs_ssa_45);
  const Float<8, 24> jfs_ssa_47 = x3.mul(JFS_RM_RNE, jfs_ssa_46);
  const Float<8, 24> jfs_ssa_48 = jfs_ssa_43.add(JFS_RM_RNE, jfs_ssa_47);
  const BitVector<23> jfs_ssa_49 = BitVector<23>(UINT64_C(2030043));
  const Float<8, 24> jfs_ssa_50 =
      Float<8, 24>(jfs_ssa_0, jfs_ssa_30, jfs_ssa_49);
  const Float<8, 24> jfs_ssa_51 = x4.mul(JFS_RM_RNE, jfs_ssa_50);
  const Float<8, 24> jfs_ssa_52 = jfs_ssa_48.add(JFS_RM_RNE, jfs_ssa_51);
  const BitVector<8> jfs_ssa_53 = BitVector<8>(UINT64_C(122));
  const BitVector<23> jfs_ssa_54 = BitVector<23>(UINT64_C(5838471));
  const Float<8, 24> jfs_ssa_55 =
      Float<8, 24>(jfs_ssa_0, jfs_ssa_53, jfs_ssa_54);
  const Float<8, 24> jfs_ssa_56 = x5.mul(JFS_RM_RNE, jfs_ssa_55);
  const Float<8, 24> jfs_ssa_57 = jfs_ssa_52.add(JFS_RM_RNE, jfs_ssa_56);
  const BitVector<23> jfs_ssa_58 = BitVector<23>(UINT64_C(1442841));
  const Float<8, 24> jfs_ssa_59 =
      Float<8, 24>(jfs_ssa_5, jfs_ssa_35, jfs_ssa_58);
  const Float<8, 24> jfs_ssa_60 = x6.mul(JFS_RM_RNE, jfs_ssa_59);
  const Float<8, 24> jfs_ssa_61 = jfs_ssa_57.add(JFS_RM_RNE, jfs_ssa_60);
  const BitVector<8> jfs_ssa_62 = BitVector<8>(UINT64_C(124));
  const BitVector<23> jfs_ssa_63 = BitVector<23>(UINT64_C(5100274));
  const Float<8, 24> jfs_ssa_64 =
      Float<8, 24>(jfs_ssa_5, jfs_ssa_62, jfs_ssa_63);
  const bool jfs_ssa_65 = jfs_ssa_61.fpleq(jfs_ssa_64);
  if (jfs_ssa_65) {
  } else {
    return 0;
  }
  const BitVector<23> jfs_ssa_66 = BitVector<23>(UINT64_C(1275067));
  const Float<8, 24> jfs_ssa_67 =
      Float<8, 24>(jfs_ssa_5, jfs_ssa_53, jfs_ssa_66);
  const BitVector<23> jfs_ssa_68 = BitVector<23>(UINT64_C(6341788));
  const Float<8, 24> jfs_ssa_69 =
      Float<8, 24>(jfs_ssa_5, jfs_ssa_30, jfs_ssa_68);
  const Float<8, 24> jfs_ssa_70 = x0.mul(JFS_RM_RNE, jfs_ssa_69);
  const Float<8, 24> jfs_ssa_71 = jfs_ssa_29.add(JFS_RM_RNE, jfs_ssa_70);
  const BitVector<23> jfs_ssa_72 = BitVector<23>(UINT64_C(3959422));
  const Float<8, 24> jfs_ssa_73 =
      Float<8, 24>(jfs_ssa_0, jfs_ssa_62, jfs_ssa_72);
  const Float<8, 24> jfs_ssa_74 = x1.mul(JFS_RM_RNE, jfs_ssa_73);
  const Float<8, 24> jfs_ssa_75 = jfs_ssa_71.add(JFS_RM_RNE, jfs_ssa_74);
  const BitVector<23> jfs_ssa_76 = BitVector<23>(UINT64_C(3137339));
  const Float<8, 24> jfs_ssa_77 =
      Float<8, 24>(jfs_ssa_5, jfs_ssa_30, jfs_ssa_76);
  const Float<8, 24> jfs_ssa_78 = x2.mul(JFS_RM_RNE, jfs_ssa_77);
  const Float<8, 24> jfs_ssa_79 = jfs_ssa_75.add(JFS_RM_RNE, jfs_ssa_78);
  const BitVector<23> jfs_ssa_80 = BitVector<23>(UINT64_C(6207569));
  const Float<8, 24> jfs_ssa_81 =
      Float<8, 24>(jfs_ssa_0, jfs_ssa_30, jfs_ssa_80);
  const Float<8, 24> jfs_ssa_82 = x3.mul(JFS_RM_RNE, jfs_ssa_81);
  const Float<8, 24> jfs_ssa_83 = jfs_ssa_79.add(JFS_RM_RNE, jfs_ssa_82);
  const BitVector<23> jfs_ssa_84 = BitVector<23>(UINT64_C(3925868));
  const Float<8, 24> jfs_ssa_85 =
      Float<8, 24>(jfs_ssa_0, jfs_ssa_30, jfs_ssa_84);
  const Float<8, 24> jfs_ssa_86 = x4.mul(JFS_RM_RNE, jfs_ssa_85);
  const Float<8, 24> jfs_ssa_87 = jfs_ssa_83.add(JFS_RM_RNE, jfs_ssa_86);
  const BitVector<23> jfs_ssa_88 = BitVector<23>(UINT64_C(7381974));
  const Float<8, 24> jfs_ssa_89 =
      Float<8, 24>(jfs_ssa_0, jfs_ssa_35, jfs_ssa_88);
  const Float<8, 24> jfs_ssa_90 = x5.mul(JFS_RM_RNE, jfs_ssa_89);
  const Float<8, 24> jfs_ssa_91 = jfs_ssa_87.add(JFS_RM_RNE, jfs_ssa_90);
  const BitVector<23> jfs_ssa_92 = BitVector<23>(UINT64_C(67109));
  const Float<8, 24> jfs_ssa_93 =
      Float<8, 24>(jfs_ssa_5, jfs_ssa_35, jfs_ssa_92);
  const Float<8, 24> jfs_ssa_94 = x6.mul(JFS_RM_RNE, jfs_ssa_93);
  const Float<8, 24> jfs_ssa_95 = jfs_ssa_91.add(JFS_RM_RNE, jfs_ssa_94);
  const bool jfs_ssa_96 = jfs_ssa_67.fpleq(jfs_ssa_95);
  if (jfs_ssa_96) {
  } else {
    return 0;
  }
  const BitVector<23> jfs_ssa_97 = BitVector<23>(UINT64_C(2365587));
  const Float<8, 24> jfs_ssa_98 =
      Float<8, 24>(jfs_ssa_5, jfs_ssa_30, jfs_ssa_97);
  const Float<8, 24> jfs_ssa_99 = x0.mul(JFS_RM_RNE, jfs_ssa_98);
  const Float<8, 24> jfs_ssa_100 = jfs_ssa_29.add(JFS_RM_RNE, jfs_ssa_99);
  const BitVector<23> jfs_ssa_101 = BitVector<23>(UINT64_C(2902457));
  const Float<8, 24> jfs_ssa_102 =
      Float<8, 24>(jfs_ssa_0, jfs_ssa_30, jfs_ssa_101);
  const Float<8, 24> jfs_ssa_103 = x1.mul(JFS_RM_RNE, jfs_ssa_102);
  const Float<8, 24> jfs_ssa_104 = jfs_ssa_100.add(JFS_RM_RNE, jfs_ssa_103);
  const BitVector<23> jfs_ssa_105 = BitVector<23>(UINT64_C(4898947));
  const Float<8, 24> jfs_ssa_106 =
      Float<8, 24>(jfs_ssa_0, jfs_ssa_62, jfs_ssa_105);
  const Float<8, 24> jfs_ssa_107 = x2.mul(JFS_RM_RNE, jfs_ssa_106);
  const Float<8, 24> jfs_ssa_108 = jfs_ssa_104.add(JFS_RM_RNE, jfs_ssa_107);
  const BitVector<23> jfs_ssa_109 = BitVector<23>(UINT64_C(369098));
  const Float<8, 24> jfs_ssa_110 =
      Float<8, 24>(jfs_ssa_5, jfs_ssa_35, jfs_ssa_109);
  const Float<8, 24> jfs_ssa_111 = x3.mul(JFS_RM_RNE, jfs_ssa_110);
  const Float<8, 24> jfs_ssa_112 = jfs_ssa_108.add(JFS_RM_RNE, jfs_ssa_111);
  const BitVector<23> jfs_ssa_113 = BitVector<23>(UINT64_C(4966056));
  const Float<8, 24> jfs_ssa_114 =
      Float<8, 24>(jfs_ssa_5, jfs_ssa_30, jfs_ssa_113);
  const Float<8, 24> jfs_ssa_115 = x4.mul(JFS_RM_RNE, jfs_ssa_114);
  const Float<8, 24> jfs_ssa_116 = jfs_ssa_112.add(JFS_RM_RNE, jfs_ssa_115);
  const BitVector<23> jfs_ssa_117 = BitVector<23>(UINT64_C(6090129));
  const Float<8, 24> jfs_ssa_118 =
      Float<8, 24>(jfs_ssa_5, jfs_ssa_30, jfs_ssa_117);
  const Float<8, 24> jfs_ssa_119 = x5.mul(JFS_RM_RNE, jfs_ssa_118);
  const Float<8, 24> jfs_ssa_120 = jfs_ssa_116.add(JFS_RM_RNE, jfs_ssa_119);
  const BitVector<23> jfs_ssa_121 = BitVector<23>(UINT64_C(2885680));
  const Float<8, 24> jfs_ssa_122 =
      Float<8, 24>(jfs_ssa_0, jfs_ssa_62, jfs_ssa_121);
  const Float<8, 24> jfs_ssa_123 = x6.mul(JFS_RM_RNE, jfs_ssa_122);
  const Float<8, 24> jfs_ssa_124 = jfs_ssa_120.add(JFS_RM_RNE, jfs_ssa_123);
  const BitVector<23> jfs_ssa_125 = BitVector<23>(UINT64_C(8153726));
  const Float<8, 24> jfs_ssa_126 =
      Float<8, 24>(jfs_ssa_5, jfs_ssa_30, jfs_ssa_125);
  const bool jfs_ssa_127 = jfs_ssa_124.fpleq(jfs_ssa_126);
  if (jfs_ssa_127) {
  } else {
    return 0;
  }
  const BitVector<23> jfs_ssa_128 = BitVector<23>(UINT64_C(637534));
  const Float<8, 24> jfs_ssa_129 =
      Float<8, 24>(jfs_ssa_5, jfs_ssa_30, jfs_ssa_128);
  const BitVector<23> jfs_ssa_130 = BitVector<23>(UINT64_C(7784628));
  const Float<8, 24> jfs_ssa_131 =
      Float<8, 24>(jfs_ssa_5, jfs_ssa_30, jfs_ssa_130);
  const Float<8, 24> jfs_ssa_132 = x0.mul(JFS_RM_RNE, jfs_ssa_131);
  const Float<8, 24> jfs_ssa_133 = jfs_ssa_29.add(JFS_RM_RNE, jfs_ssa_132);
  const BitVector<23> jfs_ssa_134 = BitVector<23>(UINT64_C(3674209));
  const Float<8, 24> jfs_ssa_135 =
      Float<8, 24>(jfs_ssa_0, jfs_ssa_30, jfs_ssa_134);
  const Float<8, 24> jfs_ssa_136 = x1.mul(JFS_RM_RNE, jfs_ssa_135);
  const Float<8, 24> jfs_ssa_137 = jfs_ssa_133.add(JFS_RM_RNE, jfs_ssa_136);
  const BitVector<23> jfs_ssa_138 = BitVector<23>(UINT64_C(5385486));
  const Float<8, 24> jfs_ssa_139 =
      Float<8, 24>(jfs_ssa_0, jfs_ssa_30, jfs_ssa_138);
  const Float<8, 24> jfs_ssa_140 = x2.mul(JFS_RM_RNE, jfs_ssa_139);
  const Float<8, 24> jfs_ssa_141 = jfs_ssa_137.add(JFS_RM_RNE, jfs_ssa_140);
  const BitVector<23> jfs_ssa_142 = BitVector<23>(UINT64_C(8120172));
  const Float<8, 24> jfs_ssa_143 =
      Float<8, 24>(jfs_ssa_5, jfs_ssa_35, jfs_ssa_142);
  const Float<8, 24> jfs_ssa_144 = x3.mul(JFS_RM_RNE, jfs_ssa_143);
  const Float<8, 24> jfs_ssa_145 = jfs_ssa_141.add(JFS_RM_RNE, jfs_ssa_144);
  const Float<8, 24> jfs_ssa_146 =
      Float<8, 24>(jfs_ssa_0, jfs_ssa_30, jfs_ssa_117);
  const Float<8, 24> jfs_ssa_147 = x4.mul(JFS_RM_RNE, jfs_ssa_146);
  const Float<8, 24> jfs_ssa_148 = jfs_ssa_145.add(JFS_RM_RNE, jfs_ssa_147);
  const BitVector<23> jfs_ssa_149 = BitVector<23>(UINT64_C(7499416));
  const Float<8, 24> jfs_ssa_150 =
      Float<8, 24>(jfs_ssa_5, jfs_ssa_30, jfs_ssa_149);
  const Float<8, 24> jfs_ssa_151 = x5.mul(JFS_RM_RNE, jfs_ssa_150);
  const Float<8, 24> jfs_ssa_152 = jfs_ssa_148.add(JFS_RM_RNE, jfs_ssa_151);
  const BitVector<23> jfs_ssa_153 = BitVector<23>(UINT64_C(3992977));
  const Float<8, 24> jfs_ssa_154 =
      Float<8, 24>(jfs_ssa_0, jfs_ssa_30, jfs_ssa_153);
  const Float<8, 24> jfs_ssa_155 = x6.mul(JFS_RM_RNE, jfs_ssa_154);
  const Float<8, 24> jfs_ssa_156 = jfs_ssa_152.add(JFS_RM_RNE, jfs_ssa_155);
  const bool jfs_ssa_157 = jfs_ssa_129.fpleq(jfs_ssa_156);
  if (jfs_ssa_157) {
  } else {
    return 0;
  }
  const BitVector<23> jfs_ssa_158 = BitVector<23>(UINT64_C(4294966));
  const Float<8, 24> jfs_ssa_159 =
      Float<8, 24>(jfs_ssa_0, jfs_ssa_35, jfs_ssa_158);
  const Float<8, 24> jfs_ssa_160 = x0.mul(JFS_RM_RNE, jfs_ssa_159);
  const Float<8, 24> jfs_ssa_161 = jfs_ssa_29.add(JFS_RM_RNE, jfs_ssa_160);
  const BitVector<23> jfs_ssa_162 = BitVector<23>(UINT64_C(4848614));
  const Float<8, 24> jfs_ssa_163 =
      Float<8, 24>(jfs_ssa_5, jfs_ssa_30, jfs_ssa_162);
  const Float<8, 24> jfs_ssa_164 = x1.mul(JFS_RM_RNE, jfs_ssa_163);
  const Float<8, 24> jfs_ssa_165 = jfs_ssa_161.add(JFS_RM_RNE, jfs_ssa_164);
  const BitVector<23> jfs_ssa_166 = BitVector<23>(UINT64_C(3623879));
  const Float<8, 24> jfs_ssa_167 =
      Float<8, 24>(jfs_ssa_5, jfs_ssa_62, jfs_ssa_166);
  const Float<8, 24> jfs_ssa_168 = x2.mul(JFS_RM_RNE, jfs_ssa_167);
  const Float<8, 24> jfs_ssa_169 = jfs_ssa_165.add(JFS_RM_RNE, jfs_ssa_168);
  const BitVector<23> jfs_ssa_170 = BitVector<23>(UINT64_C(2013266));
  const Float<8, 24> jfs_ssa_171 =
      Float<8, 24>(jfs_ssa_5, jfs_ssa_62, jfs_ssa_170);
  const Float<8, 24> jfs_ssa_172 = x3.mul(JFS_RM_RNE, jfs_ssa_171);
  const Float<8, 24> jfs_ssa_173 = jfs_ssa_169.add(JFS_RM_RNE, jfs_ssa_172);
  const BitVector<23> jfs_ssa_174 = BitVector<23>(UINT64_C(1275068));
  const Float<8, 24> jfs_ssa_175 =
      Float<8, 24>(jfs_ssa_5, jfs_ssa_35, jfs_ssa_174);
  const Float<8, 24> jfs_ssa_176 = x4.mul(JFS_RM_RNE, jfs_ssa_175);
  const Float<8, 24> jfs_ssa_177 = jfs_ssa_173.add(JFS_RM_RNE, jfs_ssa_176);
  const BitVector<23> jfs_ssa_178 = BitVector<23>(UINT64_C(7734297));
  const Float<8, 24> jfs_ssa_179 =
      Float<8, 24>(jfs_ssa_5, jfs_ssa_30, jfs_ssa_178);
  const Float<8, 24> jfs_ssa_180 = x5.mul(JFS_RM_RNE, jfs_ssa_179);
  const Float<8, 24> jfs_ssa_181 = jfs_ssa_177.add(JFS_RM_RNE, jfs_ssa_180);
  const BitVector<23> jfs_ssa_182 = BitVector<23>(UINT64_C(1342177));
  const Float<8, 24> jfs_ssa_183 =
      Float<8, 24>(jfs_ssa_0, jfs_ssa_30, jfs_ssa_182);
  const Float<8, 24> jfs_ssa_184 = x6.mul(JFS_RM_RNE, jfs_ssa_183);
  const Float<8, 24> jfs_ssa_185 = jfs_ssa_181.add(JFS_RM_RNE, jfs_ssa_184);
  const BitVector<23> jfs_ssa_186 = BitVector<23>(UINT64_C(7818182));
  const Float<8, 24> jfs_ssa_187 =
      Float<8, 24>(jfs_ssa_0, jfs_ssa_30, jfs_ssa_186);
  const bool jfs_ssa_188 = jfs_ssa_185.fpleq(jfs_ssa_187);
  if (jfs_ssa_188) {
  } else {
    return 0;
  }
  const BitVector<23> jfs_ssa_189 = BitVector<23>(UINT64_C(3154117));
  const Float<8, 24> jfs_ssa_190 =
      Float<8, 24>(jfs_ssa_0, jfs_ssa_35, jfs_ssa_189);
  const BitVector<23> jfs_ssa_191 = BitVector<23>(UINT64_C(7096761));
  const Float<8, 24> jfs_ssa_192 =
      Float<8, 24>(jfs_ssa_5, jfs_ssa_30, jfs_ssa_191);
  const Float<8, 24> jfs_ssa_193 = x0.mul(JFS_RM_RNE, jfs_ssa_192);
  const Float<8, 24> jfs_ssa_194 = jfs_ssa_29.add(JFS_RM_RNE, jfs_ssa_193);
  const BitVector<23> jfs_ssa_195 = BitVector<23>(UINT64_C(8321499));
  const Float<8, 24> jfs_ssa_196 =
      Float<8, 24>(jfs_ssa_5, jfs_ssa_62, jfs_ssa_195);
  const Float<8, 24> jfs_ssa_197 = x1.mul(JFS_RM_RNE, jfs_ssa_196);
  const Float<8, 24> jfs_ssa_198 = jfs_ssa_194.add(JFS_RM_RNE, jfs_ssa_197);
  const BitVector<23> jfs_ssa_199 = BitVector<23>(UINT64_C(469761));
  const Float<8, 24> jfs_ssa_200 =
      Float<8, 24>(jfs_ssa_5, jfs_ssa_30, jfs_ssa_199);
  const Float<8, 24> jfs_ssa_201 = x2.mul(JFS_RM_RNE, jfs_ssa_200);
  const Float<8, 24> jfs_ssa_202 = jfs_ssa_198.add(JFS_RM_RNE, jfs_ssa_201);
  const BitVector<23> jfs_ssa_203 = BitVector<23>(UINT64_C(7113540));
  const Float<8, 24> jfs_ssa_204 =
      Float<8, 24>(jfs_ssa_0, jfs_ssa_30, jfs_ssa_203);
  const Float<8, 24> jfs_ssa_205 = x3.mul(JFS_RM_RNE, jfs_ssa_204);
  const Float<8, 24> jfs_ssa_206 = jfs_ssa_202.add(JFS_RM_RNE, jfs_ssa_205);
  const BitVector<23> jfs_ssa_207 = BitVector<23>(UINT64_C(1879047));
  const Float<8, 24> jfs_ssa_208 =
      Float<8, 24>(jfs_ssa_5, jfs_ssa_62, jfs_ssa_207);
  const Float<8, 24> jfs_ssa_209 = x4.mul(JFS_RM_RNE, jfs_ssa_208);
  const Float<8, 24> jfs_ssa_210 = jfs_ssa_206.add(JFS_RM_RNE, jfs_ssa_209);
  const BitVector<23> jfs_ssa_211 = BitVector<23>(UINT64_C(5301600));
  const Float<8, 24> jfs_ssa_212 =
      Float<8, 24>(jfs_ssa_0, jfs_ssa_35, jfs_ssa_211);
  const Float<8, 24> jfs_ssa_213 = x5.mul(JFS_RM_RNE, jfs_ssa_212);
  const Float<8, 24> jfs_ssa_214 = jfs_ssa_210.add(JFS_RM_RNE, jfs_ssa_213);
  const BitVector<23> jfs_ssa_215 = BitVector<23>(UINT64_C(3640656));
  const Float<8, 24> jfs_ssa_216 =
      Float<8, 24>(jfs_ssa_5, jfs_ssa_30, jfs_ssa_215);
  const Float<8, 24> jfs_ssa_217 = x6.mul(JFS_RM_RNE, jfs_ssa_216);
  const Float<8, 24> jfs_ssa_218 = jfs_ssa_214.add(JFS_RM_RNE, jfs_ssa_217);
  const bool jfs_ssa_219 = jfs_ssa_190.fpleq(jfs_ssa_218);
  if (jfs_ssa_219) {
  } else {
    return 0;
  }
  const BitVector<23> jfs_ssa_220 = BitVector<23>(UINT64_C(7247756));
  const Float<8, 24> jfs_ssa_221 =
      Float<8, 24>(jfs_ssa_5, jfs_ssa_35, jfs_ssa_220);
  const BitVector<23> jfs_ssa_222 = BitVector<23>(UINT64_C(201326));
  const Float<8, 24> jfs_ssa_223 =
      Float<8, 24>(jfs_ssa_5, jfs_ssa_30, jfs_ssa_222);
  const Float<8, 24> jfs_ssa_224 = x0.mul(JFS_RM_RNE, jfs_ssa_223);
  const Float<8, 24> jfs_ssa_225 = jfs_ssa_29.add(JFS_RM_RNE, jfs_ssa_224);
  const BitVector<23> jfs_ssa_226 = BitVector<23>(UINT64_C(4362076));
  const Float<8, 24> jfs_ssa_227 =
      Float<8, 24>(jfs_ssa_5, jfs_ssa_44, jfs_ssa_226);
  const Float<8, 24> jfs_ssa_228 = x1.mul(JFS_RM_RNE, jfs_ssa_227);
  const Float<8, 24> jfs_ssa_229 = jfs_ssa_225.add(JFS_RM_RNE, jfs_ssa_228);
  const BitVector<23> jfs_ssa_230 = BitVector<23>(UINT64_C(4227858));
  const Float<8, 24> jfs_ssa_231 =
      Float<8, 24>(jfs_ssa_5, jfs_ssa_35, jfs_ssa_230);
  const Float<8, 24> jfs_ssa_232 = x2.mul(JFS_RM_RNE, jfs_ssa_231);
  const Float<8, 24> jfs_ssa_233 = jfs_ssa_229.add(JFS_RM_RNE, jfs_ssa_232);
  const BitVector<23> jfs_ssa_234 = BitVector<23>(UINT64_C(3204448));
  const Float<8, 24> jfs_ssa_235 =
      Float<8, 24>(jfs_ssa_5, jfs_ssa_30, jfs_ssa_234);
  const Float<8, 24> jfs_ssa_236 = x3.mul(JFS_RM_RNE, jfs_ssa_235);
  const Float<8, 24> jfs_ssa_237 = jfs_ssa_233.add(JFS_RM_RNE, jfs_ssa_236);
  const BitVector<23> jfs_ssa_238 = BitVector<23>(UINT64_C(2214593));
  const Float<8, 24> jfs_ssa_239 =
      Float<8, 24>(jfs_ssa_5, jfs_ssa_30, jfs_ssa_238);
  const Float<8, 24> jfs_ssa_240 = x4.mul(JFS_RM_RNE, jfs_ssa_239);
  const Float<8, 24> jfs_ssa_241 = jfs_ssa_237.add(JFS_RM_RNE, jfs_ssa_240);
  const BitVector<23> jfs_ssa_242 = BitVector<23>(UINT64_C(352321));
  const Float<8, 24> jfs_ssa_243 =
      Float<8, 24>(jfs_ssa_0, jfs_ssa_30, jfs_ssa_242);
  const Float<8, 24> jfs_ssa_244 = x5.mul(JFS_RM_RNE, jfs_ssa_243);
  const Float<8, 24> jfs_ssa_245 = jfs_ssa_241.add(JFS_RM_RNE, jfs_ssa_244);
  const BitVector<23> jfs_ssa_246 = BitVector<23>(UINT64_C(3120561));
  const Float<8, 24> jfs_ssa_247 =
      Float<8, 24>(jfs_ssa_5, jfs_ssa_30, jfs_ssa_246);
  const Float<8, 24> jfs_ssa_248 = x6.mul(JFS_RM_RNE, jfs_ssa_247);
  const Float<8, 24> jfs_ssa_249 = jfs_ssa_245.add(JFS_RM_RNE, jfs_ssa_248);
  const bool jfs_ssa_250 = jfs_ssa_221.fpleq(jfs_ssa_249);
  if (jfs_ssa_250) {
  } else {
    return 0;
  }
  // Fuzzing target
  return 1;
}
// End program
