from petastorm.unischema import Unischema, UnischemaField
from petastorm.codecs import ScalarCodec, NdarrayCodec
from pyspark.sql.types import IntegerType, StringType

import numpy as np

L1bSchema = Unischema('L1bSchema', [
   UnischemaField('year', np.int32, (), ScalarCodec(IntegerType()), False),
   UnischemaField('dayofyear', np.int32, (), ScalarCodec(IntegerType()), False),
   UnischemaField('hour', np.int32, (), ScalarCodec(IntegerType()), False),
   UnischemaField('minute', np.int32, (), ScalarCodec(IntegerType()), False),
   UnischemaField('file', np.string_, (), ScalarCodec(StringType()), False),
   UnischemaField('h', np.string_, (), ScalarCodec(IntegerType()), False),
   UnischemaField('v', np.string_, (), ScalarCodec(IntegerType()), False),
   UnischemaField('sample_id', np.int32, (), ScalarCodec(IntegerType()), False),
   UnischemaField('data', np.float32, (64, 64, 16), NdarrayCodec(), False),

])


MAIACSchema = Unischema('MAIAC', [
   UnischemaField('year', np.int32, (), ScalarCodec(IntegerType()), False),
   UnischemaField('dayofyear', np.int32, (), ScalarCodec(IntegerType()), False),
   UnischemaField('hour', np.int32, (), ScalarCodec(IntegerType()), False),
   UnischemaField('minute', np.int32, (), ScalarCodec(IntegerType()), False),
   UnischemaField('fileahi05', np.string_, (), ScalarCodec(StringType()), False),
   UnischemaField('fileahi12', np.string_, (), ScalarCodec(StringType()), False),
   UnischemaField('h', np.string_, (), ScalarCodec(IntegerType()), False),
   UnischemaField('v', np.string_, (), ScalarCodec(IntegerType()), False),
   UnischemaField('sample_id', np.int32, (), ScalarCodec(IntegerType()), False),
   UnischemaField('AHI05', np.float32, (64, 64, 16), NdarrayCodec(), False),
   UnischemaField('AHI12', np.float32, (64, 64, 6), NdarrayCodec(), False),
])

MAIACSchema256 = Unischema('MAIAC256', [
   UnischemaField('year', np.int32, (), ScalarCodec(IntegerType()), False),
   UnischemaField('dayofyear', np.int32, (), ScalarCodec(IntegerType()), False),
   UnischemaField('hour', np.int32, (), ScalarCodec(IntegerType()), False),
   UnischemaField('minute', np.int32, (), ScalarCodec(IntegerType()), False),
   UnischemaField('fileahi05', np.string_, (), ScalarCodec(StringType()), False),
   UnischemaField('fileahi12', np.string_, (), ScalarCodec(StringType()), False),
   UnischemaField('h', np.string_, (), ScalarCodec(IntegerType()), False),
   UnischemaField('v', np.string_, (), ScalarCodec(IntegerType()), False),
   UnischemaField('sample_id', np.int32, (), ScalarCodec(IntegerType()), False),
   UnischemaField('AHI05', np.float32, (256, 256, 16), NdarrayCodec(), False),
   UnischemaField('AHI12', np.float32, (256, 256, 6), NdarrayCodec(), False),
])
