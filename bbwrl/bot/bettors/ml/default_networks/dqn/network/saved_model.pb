Ԋ
?
?

B
AssignVariableOp
resource
value"dtype"
dtypetype?
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.4.12v2.4.0-49-g85c8b2a817f8??
t
mlp/linear_0/bVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namemlp/linear_0/b
m
"mlp/linear_0/b/Read/ReadVariableOpReadVariableOpmlp/linear_0/b*
_output_shapes
: *
dtype0
x
mlp/linear_0/wVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *
shared_namemlp/linear_0/w
q
"mlp/linear_0/w/Read/ReadVariableOpReadVariableOpmlp/linear_0/w*
_output_shapes

: *
dtype0
t
mlp/linear_1/bVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namemlp/linear_1/b
m
"mlp/linear_1/b/Read/ReadVariableOpReadVariableOpmlp/linear_1/b*
_output_shapes
:*
dtype0
x
mlp/linear_1/wVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *
shared_namemlp/linear_1/w
q
"mlp/linear_1/w/Read/ReadVariableOpReadVariableOpmlp/linear_1/w*
_output_shapes

: *
dtype0
t
mlp/linear_2/bVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namemlp/linear_2/b
m
"mlp/linear_2/b/Read/ReadVariableOpReadVariableOpmlp/linear_2/b*
_output_shapes
:*
dtype0
x
mlp/linear_2/wVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namemlp/linear_2/w
q
"mlp/linear_2/w/Read/ReadVariableOpReadVariableOpmlp/linear_2/w*
_output_shapes

:*
dtype0
t
mlp/linear_3/bVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*
shared_namemlp/linear_3/b
m
"mlp/linear_3/b/Read/ReadVariableOpReadVariableOpmlp/linear_3/b*
_output_shapes
:	*
dtype0
x
mlp/linear_3/wVarHandleOp*
_output_shapes
: *
dtype0*
shape
:	*
shared_namemlp/linear_3/w
q
"mlp/linear_3/w/Read/ReadVariableOpReadVariableOpmlp/linear_3/w*
_output_shapes

:	*
dtype0

NoOpNoOp
?
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?
value?B? B?
:

_variables
_trainable_variables

signatures
8
0
1
2
3
4
	5

6
7
8
0
1
2
3
4
	5

6
7
 
KI
VARIABLE_VALUEmlp/linear_0/b'_variables/0/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEmlp/linear_0/w'_variables/1/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEmlp/linear_1/b'_variables/2/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEmlp/linear_1/w'_variables/3/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEmlp/linear_2/b'_variables/4/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEmlp/linear_2/w'_variables/5/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEmlp/linear_3/b'_variables/6/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEmlp/linear_3/w'_variables/7/.ATTRIBUTES/VARIABLE_VALUE
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCallStatefulPartitionedCallsaver_filename"mlp/linear_0/b/Read/ReadVariableOp"mlp/linear_0/w/Read/ReadVariableOp"mlp/linear_1/b/Read/ReadVariableOp"mlp/linear_1/w/Read/ReadVariableOp"mlp/linear_2/b/Read/ReadVariableOp"mlp/linear_2/w/Read/ReadVariableOp"mlp/linear_3/b/Read/ReadVariableOp"mlp/linear_3/w/Read/ReadVariableOpConst*
Tin
2
*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? **
f%R#
!__inference__traced_save_42939347
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamemlp/linear_0/bmlp/linear_0/wmlp/linear_1/bmlp/linear_1/wmlp/linear_2/bmlp/linear_2/wmlp/linear_3/bmlp/linear_3/w*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *-
f(R&
$__inference__traced_restore_42939381??
?$
?
$__inference__traced_restore_42939381
file_prefix#
assignvariableop_mlp_linear_0_b%
!assignvariableop_1_mlp_linear_0_w%
!assignvariableop_2_mlp_linear_1_b%
!assignvariableop_3_mlp_linear_1_w%
!assignvariableop_4_mlp_linear_2_b%
!assignvariableop_5_mlp_linear_2_w%
!assignvariableop_6_mlp_linear_3_b%
!assignvariableop_7_mlp_linear_3_w

identity_9??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_2?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*?
value?B?	B'_variables/0/.ATTRIBUTES/VARIABLE_VALUEB'_variables/1/.ATTRIBUTES/VARIABLE_VALUEB'_variables/2/.ATTRIBUTES/VARIABLE_VALUEB'_variables/3/.ATTRIBUTES/VARIABLE_VALUEB'_variables/4/.ATTRIBUTES/VARIABLE_VALUEB'_variables/5/.ATTRIBUTES/VARIABLE_VALUEB'_variables/6/.ATTRIBUTES/VARIABLE_VALUEB'_variables/7/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*%
valueB	B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*8
_output_shapes&
$:::::::::*
dtypes
2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOpassignvariableop_mlp_linear_0_bIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp!assignvariableop_1_mlp_linear_0_wIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp!assignvariableop_2_mlp_linear_1_bIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp!assignvariableop_3_mlp_linear_1_wIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp!assignvariableop_4_mlp_linear_2_bIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp!assignvariableop_5_mlp_linear_2_wIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp!assignvariableop_6_mlp_linear_3_bIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp!assignvariableop_7_mlp_linear_3_wIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_79
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?

Identity_8Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_8?

Identity_9IdentityIdentity_8:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7*
T0*
_output_shapes
: 2

Identity_9"!

identity_9Identity_9:output:0*5
_input_shapes$
": ::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_7:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?D
?
__inference_wrapped_module_5037

args_0
args_0_1/
+mlp_linear_0_matmul_readvariableop_resource,
(mlp_linear_0_add_readvariableop_resource/
+mlp_linear_1_matmul_readvariableop_resource,
(mlp_linear_1_add_readvariableop_resource/
+mlp_linear_2_matmul_readvariableop_resource,
(mlp_linear_2_add_readvariableop_resource/
+mlp_linear_3_matmul_readvariableop_resource,
(mlp_linear_3_add_readvariableop_resource
identity??mlp/linear_0/Add/ReadVariableOp?"mlp/linear_0/MatMul/ReadVariableOp?mlp/linear_1/Add/ReadVariableOp?"mlp/linear_1/MatMul/ReadVariableOp?mlp/linear_2/Add/ReadVariableOp?"mlp/linear_2/MatMul/ReadVariableOp?mlp/linear_3/Add/ReadVariableOp?"mlp/linear_3/MatMul/ReadVariableOpj
sequential/flatten/ShapeShapeargs_0*
T0*
_output_shapes
:2
sequential/flatten/Shape?
&sequential/flatten/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential/flatten/strided_slice/stack?
(sequential/flatten/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(sequential/flatten/strided_slice/stack_1?
(sequential/flatten/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(sequential/flatten/strided_slice/stack_2?
 sequential/flatten/strided_sliceStridedSlice!sequential/flatten/Shape:output:0/sequential/flatten/strided_slice/stack:output:01sequential/flatten/strided_slice/stack_1:output:01sequential/flatten/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2"
 sequential/flatten/strided_slice?
"sequential/flatten/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"sequential/flatten/concat/values_1?
sequential/flatten/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
sequential/flatten/concat/axis?
sequential/flatten/concatConcatV2)sequential/flatten/strided_slice:output:0+sequential/flatten/concat/values_1:output:0'sequential/flatten/concat/axis:output:0*
N*
T0*
_output_shapes
:2
sequential/flatten/concat?
sequential/flatten/ReshapeReshapeargs_0"sequential/flatten/concat:output:0*
T0*'
_output_shapes
:?????????2
sequential/flatten/Reshapep
sequential/flatten/Shape_1Shapeargs_0_1*
T0*
_output_shapes
:2
sequential/flatten/Shape_1?
(sequential/flatten/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(sequential/flatten/strided_slice_1/stack?
*sequential/flatten/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*sequential/flatten/strided_slice_1/stack_1?
*sequential/flatten/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*sequential/flatten/strided_slice_1/stack_2?
"sequential/flatten/strided_slice_1StridedSlice#sequential/flatten/Shape_1:output:01sequential/flatten/strided_slice_1/stack:output:03sequential/flatten/strided_slice_1/stack_1:output:03sequential/flatten/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2$
"sequential/flatten/strided_slice_1?
$sequential/flatten/concat_1/values_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$sequential/flatten/concat_1/values_1?
 sequential/flatten/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 sequential/flatten/concat_1/axis?
sequential/flatten/concat_1ConcatV2+sequential/flatten/strided_slice_1:output:0-sequential/flatten/concat_1/values_1:output:0)sequential/flatten/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
sequential/flatten/concat_1?
sequential/flatten/Reshape_1Reshapeargs_0_1$sequential/flatten/concat_1:output:0*
T0*'
_output_shapes
:?????????2
sequential/flatten/Reshape_1{
sequential/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
sequential/concat/axis?
sequential/concatConcatV2#sequential/flatten/Reshape:output:0%sequential/flatten/Reshape_1:output:0sequential/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2
sequential/concat?
"mlp/linear_0/MatMul/ReadVariableOpReadVariableOp+mlp_linear_0_matmul_readvariableop_resource*
_output_shapes

: *
dtype02$
"mlp/linear_0/MatMul/ReadVariableOp?
mlp/linear_0/MatMulMatMulsequential/concat:output:0*mlp/linear_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
mlp/linear_0/MatMul?
mlp/linear_0/Add/ReadVariableOpReadVariableOp(mlp_linear_0_add_readvariableop_resource*
_output_shapes
: *
dtype02!
mlp/linear_0/Add/ReadVariableOp?
mlp/linear_0/AddAddmlp/linear_0/MatMul:product:0'mlp/linear_0/Add/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
mlp/linear_0/Addd
mlp/ReluRelumlp/linear_0/Add:z:0*
T0*'
_output_shapes
:????????? 2

mlp/Relu?
"mlp/linear_1/MatMul/ReadVariableOpReadVariableOp+mlp_linear_1_matmul_readvariableop_resource*
_output_shapes

: *
dtype02$
"mlp/linear_1/MatMul/ReadVariableOp?
mlp/linear_1/MatMulMatMulmlp/Relu:activations:0*mlp/linear_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
mlp/linear_1/MatMul?
mlp/linear_1/Add/ReadVariableOpReadVariableOp(mlp_linear_1_add_readvariableop_resource*
_output_shapes
:*
dtype02!
mlp/linear_1/Add/ReadVariableOp?
mlp/linear_1/AddAddmlp/linear_1/MatMul:product:0'mlp/linear_1/Add/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
mlp/linear_1/Addh

mlp/Relu_1Relumlp/linear_1/Add:z:0*
T0*'
_output_shapes
:?????????2

mlp/Relu_1?
"mlp/linear_2/MatMul/ReadVariableOpReadVariableOp+mlp_linear_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype02$
"mlp/linear_2/MatMul/ReadVariableOp?
mlp/linear_2/MatMulMatMulmlp/Relu_1:activations:0*mlp/linear_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
mlp/linear_2/MatMul?
mlp/linear_2/Add/ReadVariableOpReadVariableOp(mlp_linear_2_add_readvariableop_resource*
_output_shapes
:*
dtype02!
mlp/linear_2/Add/ReadVariableOp?
mlp/linear_2/AddAddmlp/linear_2/MatMul:product:0'mlp/linear_2/Add/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
mlp/linear_2/Addh

mlp/Relu_2Relumlp/linear_2/Add:z:0*
T0*'
_output_shapes
:?????????2

mlp/Relu_2?
"mlp/linear_3/MatMul/ReadVariableOpReadVariableOp+mlp_linear_3_matmul_readvariableop_resource*
_output_shapes

:	*
dtype02$
"mlp/linear_3/MatMul/ReadVariableOp?
mlp/linear_3/MatMulMatMulmlp/Relu_2:activations:0*mlp/linear_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	2
mlp/linear_3/MatMul?
mlp/linear_3/Add/ReadVariableOpReadVariableOp(mlp_linear_3_add_readvariableop_resource*
_output_shapes
:	*
dtype02!
mlp/linear_3/Add/ReadVariableOp?
mlp/linear_3/AddAddmlp/linear_3/MatMul:product:0'mlp/linear_3/Add/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	2
mlp/linear_3/Add?
IdentityIdentitymlp/linear_3/Add:z:0 ^mlp/linear_0/Add/ReadVariableOp#^mlp/linear_0/MatMul/ReadVariableOp ^mlp/linear_1/Add/ReadVariableOp#^mlp/linear_1/MatMul/ReadVariableOp ^mlp/linear_2/Add/ReadVariableOp#^mlp/linear_2/MatMul/ReadVariableOp ^mlp/linear_3/Add/ReadVariableOp#^mlp/linear_3/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????	2

Identity"
identityIdentity:output:0*Y
_input_shapesH
F:?????????:?????????::::::::2B
mlp/linear_0/Add/ReadVariableOpmlp/linear_0/Add/ReadVariableOp2H
"mlp/linear_0/MatMul/ReadVariableOp"mlp/linear_0/MatMul/ReadVariableOp2B
mlp/linear_1/Add/ReadVariableOpmlp/linear_1/Add/ReadVariableOp2H
"mlp/linear_1/MatMul/ReadVariableOp"mlp/linear_1/MatMul/ReadVariableOp2B
mlp/linear_2/Add/ReadVariableOpmlp/linear_2/Add/ReadVariableOp2H
"mlp/linear_2/MatMul/ReadVariableOp"mlp/linear_2/MatMul/ReadVariableOp2B
mlp/linear_3/Add/ReadVariableOpmlp/linear_3/Add/ReadVariableOp2H
"mlp/linear_3/MatMul/ReadVariableOp"mlp/linear_3/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameargs_0:OK
'
_output_shapes
:?????????
 
_user_specified_nameargs_0
?
?
!__inference__traced_save_42939347
file_prefix-
)savev2_mlp_linear_0_b_read_readvariableop-
)savev2_mlp_linear_0_w_read_readvariableop-
)savev2_mlp_linear_1_b_read_readvariableop-
)savev2_mlp_linear_1_w_read_readvariableop-
)savev2_mlp_linear_2_b_read_readvariableop-
)savev2_mlp_linear_2_w_read_readvariableop-
)savev2_mlp_linear_3_b_read_readvariableop-
)savev2_mlp_linear_3_w_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*?
value?B?	B'_variables/0/.ATTRIBUTES/VARIABLE_VALUEB'_variables/1/.ATTRIBUTES/VARIABLE_VALUEB'_variables/2/.ATTRIBUTES/VARIABLE_VALUEB'_variables/3/.ATTRIBUTES/VARIABLE_VALUEB'_variables/4/.ATTRIBUTES/VARIABLE_VALUEB'_variables/5/.ATTRIBUTES/VARIABLE_VALUEB'_variables/6/.ATTRIBUTES/VARIABLE_VALUEB'_variables/7/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*%
valueB	B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_mlp_linear_0_b_read_readvariableop)savev2_mlp_linear_0_w_read_readvariableop)savev2_mlp_linear_1_b_read_readvariableop)savev2_mlp_linear_1_w_read_readvariableop)savev2_mlp_linear_2_b_read_readvariableop)savev2_mlp_linear_2_w_read_readvariableop)savev2_mlp_linear_3_b_read_readvariableop)savev2_mlp_linear_3_w_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
2	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*W
_input_shapesF
D: : : :: :::	:	: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix: 

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::$ 

_output_shapes

: : 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
:	:$ 

_output_shapes

:	:	

_output_shapes
: 
?	
?
__inference___call___5056
args_0_card_distribution
args_0_chips
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallargs_0_card_distributionargs_0_chipsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????	**
_read_only_resource_inputs

	*0
config_proto 

CPU

GPU2*0J 8? *(
f#R!
__inference_wrapped_module_50372
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????	2

Identity"
identityIdentity:output:0*Y
_input_shapesH
F:?????????:?????????::::::::22
StatefulPartitionedCallStatefulPartitionedCall:a ]
'
_output_shapes
:?????????
2
_user_specified_nameargs_0/CARD_DISTRIBUTION:UQ
'
_output_shapes
:?????????
&
_user_specified_nameargs_0/CHIPS
?D
?
__inference_wrapped_module_5108
args_0_card_distribution
args_0_chips/
+mlp_linear_0_matmul_readvariableop_resource,
(mlp_linear_0_add_readvariableop_resource/
+mlp_linear_1_matmul_readvariableop_resource,
(mlp_linear_1_add_readvariableop_resource/
+mlp_linear_2_matmul_readvariableop_resource,
(mlp_linear_2_add_readvariableop_resource/
+mlp_linear_3_matmul_readvariableop_resource,
(mlp_linear_3_add_readvariableop_resource
identity??mlp/linear_0/Add/ReadVariableOp?"mlp/linear_0/MatMul/ReadVariableOp?mlp/linear_1/Add/ReadVariableOp?"mlp/linear_1/MatMul/ReadVariableOp?mlp/linear_2/Add/ReadVariableOp?"mlp/linear_2/MatMul/ReadVariableOp?mlp/linear_3/Add/ReadVariableOp?"mlp/linear_3/MatMul/ReadVariableOp|
sequential/flatten/ShapeShapeargs_0_card_distribution*
T0*
_output_shapes
:2
sequential/flatten/Shape?
&sequential/flatten/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential/flatten/strided_slice/stack?
(sequential/flatten/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(sequential/flatten/strided_slice/stack_1?
(sequential/flatten/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(sequential/flatten/strided_slice/stack_2?
 sequential/flatten/strided_sliceStridedSlice!sequential/flatten/Shape:output:0/sequential/flatten/strided_slice/stack:output:01sequential/flatten/strided_slice/stack_1:output:01sequential/flatten/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2"
 sequential/flatten/strided_slice?
"sequential/flatten/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"sequential/flatten/concat/values_1?
sequential/flatten/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
sequential/flatten/concat/axis?
sequential/flatten/concatConcatV2)sequential/flatten/strided_slice:output:0+sequential/flatten/concat/values_1:output:0'sequential/flatten/concat/axis:output:0*
N*
T0*
_output_shapes
:2
sequential/flatten/concat?
sequential/flatten/ReshapeReshapeargs_0_card_distribution"sequential/flatten/concat:output:0*
T0*'
_output_shapes
:?????????2
sequential/flatten/Reshapet
sequential/flatten/Shape_1Shapeargs_0_chips*
T0*
_output_shapes
:2
sequential/flatten/Shape_1?
(sequential/flatten/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(sequential/flatten/strided_slice_1/stack?
*sequential/flatten/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*sequential/flatten/strided_slice_1/stack_1?
*sequential/flatten/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*sequential/flatten/strided_slice_1/stack_2?
"sequential/flatten/strided_slice_1StridedSlice#sequential/flatten/Shape_1:output:01sequential/flatten/strided_slice_1/stack:output:03sequential/flatten/strided_slice_1/stack_1:output:03sequential/flatten/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2$
"sequential/flatten/strided_slice_1?
$sequential/flatten/concat_1/values_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$sequential/flatten/concat_1/values_1?
 sequential/flatten/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 sequential/flatten/concat_1/axis?
sequential/flatten/concat_1ConcatV2+sequential/flatten/strided_slice_1:output:0-sequential/flatten/concat_1/values_1:output:0)sequential/flatten/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
sequential/flatten/concat_1?
sequential/flatten/Reshape_1Reshapeargs_0_chips$sequential/flatten/concat_1:output:0*
T0*'
_output_shapes
:?????????2
sequential/flatten/Reshape_1{
sequential/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
sequential/concat/axis?
sequential/concatConcatV2#sequential/flatten/Reshape:output:0%sequential/flatten/Reshape_1:output:0sequential/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2
sequential/concat?
"mlp/linear_0/MatMul/ReadVariableOpReadVariableOp+mlp_linear_0_matmul_readvariableop_resource*
_output_shapes

: *
dtype02$
"mlp/linear_0/MatMul/ReadVariableOp?
mlp/linear_0/MatMulMatMulsequential/concat:output:0*mlp/linear_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
mlp/linear_0/MatMul?
mlp/linear_0/Add/ReadVariableOpReadVariableOp(mlp_linear_0_add_readvariableop_resource*
_output_shapes
: *
dtype02!
mlp/linear_0/Add/ReadVariableOp?
mlp/linear_0/AddAddmlp/linear_0/MatMul:product:0'mlp/linear_0/Add/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
mlp/linear_0/Addd
mlp/ReluRelumlp/linear_0/Add:z:0*
T0*'
_output_shapes
:????????? 2

mlp/Relu?
"mlp/linear_1/MatMul/ReadVariableOpReadVariableOp+mlp_linear_1_matmul_readvariableop_resource*
_output_shapes

: *
dtype02$
"mlp/linear_1/MatMul/ReadVariableOp?
mlp/linear_1/MatMulMatMulmlp/Relu:activations:0*mlp/linear_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
mlp/linear_1/MatMul?
mlp/linear_1/Add/ReadVariableOpReadVariableOp(mlp_linear_1_add_readvariableop_resource*
_output_shapes
:*
dtype02!
mlp/linear_1/Add/ReadVariableOp?
mlp/linear_1/AddAddmlp/linear_1/MatMul:product:0'mlp/linear_1/Add/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
mlp/linear_1/Addh

mlp/Relu_1Relumlp/linear_1/Add:z:0*
T0*'
_output_shapes
:?????????2

mlp/Relu_1?
"mlp/linear_2/MatMul/ReadVariableOpReadVariableOp+mlp_linear_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype02$
"mlp/linear_2/MatMul/ReadVariableOp?
mlp/linear_2/MatMulMatMulmlp/Relu_1:activations:0*mlp/linear_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
mlp/linear_2/MatMul?
mlp/linear_2/Add/ReadVariableOpReadVariableOp(mlp_linear_2_add_readvariableop_resource*
_output_shapes
:*
dtype02!
mlp/linear_2/Add/ReadVariableOp?
mlp/linear_2/AddAddmlp/linear_2/MatMul:product:0'mlp/linear_2/Add/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
mlp/linear_2/Addh

mlp/Relu_2Relumlp/linear_2/Add:z:0*
T0*'
_output_shapes
:?????????2

mlp/Relu_2?
"mlp/linear_3/MatMul/ReadVariableOpReadVariableOp+mlp_linear_3_matmul_readvariableop_resource*
_output_shapes

:	*
dtype02$
"mlp/linear_3/MatMul/ReadVariableOp?
mlp/linear_3/MatMulMatMulmlp/Relu_2:activations:0*mlp/linear_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	2
mlp/linear_3/MatMul?
mlp/linear_3/Add/ReadVariableOpReadVariableOp(mlp_linear_3_add_readvariableop_resource*
_output_shapes
:	*
dtype02!
mlp/linear_3/Add/ReadVariableOp?
mlp/linear_3/AddAddmlp/linear_3/MatMul:product:0'mlp/linear_3/Add/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	2
mlp/linear_3/Add?
IdentityIdentitymlp/linear_3/Add:z:0 ^mlp/linear_0/Add/ReadVariableOp#^mlp/linear_0/MatMul/ReadVariableOp ^mlp/linear_1/Add/ReadVariableOp#^mlp/linear_1/MatMul/ReadVariableOp ^mlp/linear_2/Add/ReadVariableOp#^mlp/linear_2/MatMul/ReadVariableOp ^mlp/linear_3/Add/ReadVariableOp#^mlp/linear_3/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????	2

Identity"
identityIdentity:output:0*Y
_input_shapesH
F:?????????:?????????::::::::2B
mlp/linear_0/Add/ReadVariableOpmlp/linear_0/Add/ReadVariableOp2H
"mlp/linear_0/MatMul/ReadVariableOp"mlp/linear_0/MatMul/ReadVariableOp2B
mlp/linear_1/Add/ReadVariableOpmlp/linear_1/Add/ReadVariableOp2H
"mlp/linear_1/MatMul/ReadVariableOp"mlp/linear_1/MatMul/ReadVariableOp2B
mlp/linear_2/Add/ReadVariableOpmlp/linear_2/Add/ReadVariableOp2H
"mlp/linear_2/MatMul/ReadVariableOp"mlp/linear_2/MatMul/ReadVariableOp2B
mlp/linear_3/Add/ReadVariableOpmlp/linear_3/Add/ReadVariableOp2H
"mlp/linear_3/MatMul/ReadVariableOp"mlp/linear_3/MatMul/ReadVariableOp:a ]
'
_output_shapes
:?????????
2
_user_specified_nameargs_0/CARD_DISTRIBUTION:UQ
'
_output_shapes
:?????????
&
_user_specified_nameargs_0/CHIPS"?J
saver_filename:0StatefulPartitionedCall:0StatefulPartitionedCall_18"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp:?

l

_variables
_trainable_variables

signatures
__call__
_module"
acme_snapshot
Y
0
1
2
3
4
	5

6
7"
trackable_tuple_wrapper
Y
0
1
2
3
4
	5

6
7"
trackable_tuple_wrapper
"
signature_map
: 2mlp/linear_0/b
 : 2mlp/linear_0/w
:2mlp/linear_1/b
 : 2mlp/linear_1/w
:2mlp/linear_2/b
 :2mlp/linear_2/w
:	2mlp/linear_3/b
 :	2mlp/linear_3/w
?2?
__inference___call___5056?
???
FullArgSpec
args?
jself
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
__inference_wrapped_module_5108?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
__inference___call___5056?	
???
??
}?z
G
CARD_DISTRIBUTION2?/
args_0/CARD_DISTRIBUTION?????????
/
CHIPS&?#
args_0/CHIPS?????????
? "??????????	?
__inference_wrapped_module_5108?	
???
??
}?z
G
CARD_DISTRIBUTION2?/
args_0/CARD_DISTRIBUTION?????????
/
CHIPS&?#
args_0/CHIPS?????????
? "??????????	