// Sizes, alignments, and every serialized descriptor member's byte offset.
#include "ToucanGPUGenDataTypes.h"
#include <cstddef>
#include <iostream>
using namespace toucanGPUSim;
#define TYPE(T) std::cout << #T << ' ' << sizeof(T) << ' ' << alignof(T) << '\n'
#define FIELD(T, F) std::cout << #T "." #F << ' ' << offsetof(T, F) << '\n'
int main() {
  TYPE(CGRegReadMetaInfo);
  FIELD(CGRegReadMetaInfo, reg); FIELD(CGRegReadMetaInfo, result);
  TYPE(CGExchangeReadMetaInfo);
  FIELD(CGExchangeReadMetaInfo, exchange); FIELD(CGExchangeReadMetaInfo, result);
  TYPE(CGExchangeWriteMetaInfo);
  FIELD(CGExchangeWriteMetaInfo, exchange); FIELD(CGExchangeWriteMetaInfo, dat);
  FIELD(CGExchangeWriteMetaInfo, count);
  TYPE(CGPrintMetaInfo); FIELD(CGPrintMetaInfo, en); FIELD(CGPrintMetaInfo, msg);
  TYPE(CGStopMetaInfo); FIELD(CGStopMetaInfo, en);
  TYPE(CGRegWriteMetaInfo);
  FIELD(CGRegWriteMetaInfo, reg); FIELD(CGRegWriteMetaInfo, dat); FIELD(CGRegWriteMetaInfo, count);
  TYPE(CGMemWriteMetaInfo);
  FIELD(CGMemWriteMetaInfo, hasMultipleWriter); FIELD(CGMemWriteMetaInfo, memBase);
  FIELD(CGMemWriteMetaInfo, addrVec); FIELD(CGMemWriteMetaInfo, dat); FIELD(CGMemWriteMetaInfo, en);
  TYPE(CGMicroPartLUTTopLevelOp);
  FIELD(CGMicroPartLUTTopLevelOp, lutIndex); FIELD(CGMicroPartLUTTopLevelOp, op0);
  FIELD(CGMicroPartLUTTopLevelOp, op1); FIELD(CGMicroPartLUTTopLevelOp, op2);
  TYPE(CGMicroPartLUTMiddleLevelOp);
  TYPE(CGMicroPartLUTLastLevelWriteBack);
  FIELD(CGMicroPartLUTLastLevelWriteBack, shuffleId); FIELD(CGMicroPartLUTLastLevelWriteBack, result);
  TYPE(CGMicroPartVecRead);
  FIELD(CGMicroPartVecRead, vecBase); FIELD(CGMicroPartVecRead, vecLength);
  FIELD(CGMicroPartVecRead, offset); FIELD(CGMicroPartVecRead, isConstVec);
  FIELD(CGMicroPartVecRead, index0); FIELD(CGMicroPartVecRead, index1);
  FIELD(CGMicroPartVecRead, index2); FIELD(CGMicroPartVecRead, index3);
  FIELD(CGMicroPartVecRead, outRangeValue); FIELD(CGMicroPartVecRead, result);
  TYPE(CGMicroPartVecArithOrLogic);
  FIELD(CGMicroPartVecArithOrLogic, vec1Base); FIELD(CGMicroPartVecArithOrLogic, vec2Base);
  FIELD(CGMicroPartVecArithOrLogic, vecLength); FIELD(CGMicroPartVecArithOrLogic, opName);
  FIELD(CGMicroPartVecArithOrLogic, isV1V2Const); FIELD(CGMicroPartVecArithOrLogic, result);
  TYPE(CGMicroPartMemRead);
  FIELD(CGMicroPartMemRead, hasMultipleWriter); FIELD(CGMicroPartMemRead, memBase);
  FIELD(CGMicroPartMemRead, en); FIELD(CGMicroPartMemRead, addrVec); FIELD(CGMicroPartMemRead, result);
}
