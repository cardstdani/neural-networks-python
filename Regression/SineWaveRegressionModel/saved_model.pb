??
??
D
AddV2
x"T
y"T
z"T"
Ttype:
2	??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
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
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
?
Mul
x"T
y"T
z"T"
Ttype:
2	?
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
,
Sin
x"T
y"T"
Ttype:

2
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
 ?"serve*2.5.02v2.5.0-0-ga4dfb8d1a718??
l
VariableVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_name
Variable
e
Variable/Read/ReadVariableOpReadVariableOpVariable*
_output_shapes

:*
dtype0
l

Variable_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
Variable_1
e
Variable_1/Read/ReadVariableOpReadVariableOp
Variable_1*
_output_shapes
:*
dtype0
p

Variable_2VarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_name
Variable_2
i
Variable_2/Read/ReadVariableOpReadVariableOp
Variable_2*
_output_shapes

:*
dtype0
|
dense_307/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*!
shared_namedense_307/kernel
u
$dense_307/kernel/Read/ReadVariableOpReadVariableOpdense_307/kernel*
_output_shapes

:*
dtype0
t
dense_307/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_307/bias
m
"dense_307/bias/Read/ReadVariableOpReadVariableOpdense_307/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_2
[
total_2/Read/ReadVariableOpReadVariableOptotal_2*
_output_shapes
: *
dtype0
b
count_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_2
[
count_2/Read/ReadVariableOpReadVariableOpcount_2*
_output_shapes
: *
dtype0
z
Adam/Variable/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_nameAdam/Variable/m
s
#Adam/Variable/m/Read/ReadVariableOpReadVariableOpAdam/Variable/m*
_output_shapes

:*
dtype0
z
Adam/Variable/m_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/Variable/m_1
s
%Adam/Variable/m_1/Read/ReadVariableOpReadVariableOpAdam/Variable/m_1*
_output_shapes
:*
dtype0
~
Adam/Variable/m_2VarHandleOp*
_output_shapes
: *
dtype0*
shape
:*"
shared_nameAdam/Variable/m_2
w
%Adam/Variable/m_2/Read/ReadVariableOpReadVariableOpAdam/Variable/m_2*
_output_shapes

:*
dtype0
?
Adam/dense_307/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/dense_307/kernel/m
?
+Adam/dense_307/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_307/kernel/m*
_output_shapes

:*
dtype0
?
Adam/dense_307/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_307/bias/m
{
)Adam/dense_307/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_307/bias/m*
_output_shapes
:*
dtype0
z
Adam/Variable/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_nameAdam/Variable/v
s
#Adam/Variable/v/Read/ReadVariableOpReadVariableOpAdam/Variable/v*
_output_shapes

:*
dtype0
z
Adam/Variable/v_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/Variable/v_1
s
%Adam/Variable/v_1/Read/ReadVariableOpReadVariableOpAdam/Variable/v_1*
_output_shapes
:*
dtype0
~
Adam/Variable/v_2VarHandleOp*
_output_shapes
: *
dtype0*
shape
:*"
shared_nameAdam/Variable/v_2
w
%Adam/Variable/v_2/Read/ReadVariableOpReadVariableOpAdam/Variable/v_2*
_output_shapes

:*
dtype0
?
Adam/dense_307/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/dense_307/kernel/v
?
+Adam/dense_307/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_307/kernel/v*
_output_shapes

:*
dtype0
?
Adam/dense_307/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_307/bias/v
{
)Adam/dense_307/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_307/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
?!
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*? 
value? B?  B? 
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
	optimizer
trainable_variables
regularization_losses
	variables
	keras_api

signatures
g
	w

b
a
trainable_variables
regularization_losses
	variables
	keras_api
h

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
?
iter

beta_1

beta_2
	decay
learning_rate	m;
m<m=m>m?	v@
vAvBvCvD
#
	0

1
2
3
4
 
#
	0

1
2
3
4
?

layers
layer_regularization_losses
layer_metrics
trainable_variables
non_trainable_variables
regularization_losses
metrics
	variables
 
OM
VARIABLE_VALUEVariable1layer_with_weights-0/w/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUE
Variable_11layer_with_weights-0/b/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUE
Variable_21layer_with_weights-0/a/.ATTRIBUTES/VARIABLE_VALUE

	0

1
2
 

	0

1
2
?

 layers
!layer_regularization_losses
"layer_metrics
trainable_variables
#non_trainable_variables
regularization_losses
$metrics
	variables
\Z
VARIABLE_VALUEdense_307/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_307/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
?

%layers
&layer_regularization_losses
'layer_metrics
trainable_variables
(non_trainable_variables
regularization_losses
)metrics
	variables
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE

0
1
 
 
 

*0
+1
,2
 
 
 
 
 
 
 
 
 
 
4
	-total
	.count
/	variables
0	keras_api
D
	1total
	2count
3
_fn_kwargs
4	variables
5	keras_api
D
	6total
	7count
8
_fn_kwargs
9	variables
:	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

-0
.1

/	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

10
21

4	variables
QO
VARIABLE_VALUEtotal_24keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_24keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUE
 

60
71

9	variables
rp
VARIABLE_VALUEAdam/Variable/mMlayer_with_weights-0/w/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUEAdam/Variable/m_1Mlayer_with_weights-0/b/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUEAdam/Variable/m_2Mlayer_with_weights-0/a/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_307/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_307/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEAdam/Variable/vMlayer_with_weights-0/w/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUEAdam/Variable/v_1Mlayer_with_weights-0/b/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUEAdam/Variable/v_2Mlayer_with_weights-0/a/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_307/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_307/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_sine_85_inputPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_sine_85_inputVariable
Variable_1
Variable_2dense_307/kerneldense_307/bias*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8? *.
f)R'
%__inference_signature_wrapper_2864174
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?	
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameVariable/Read/ReadVariableOpVariable_1/Read/ReadVariableOpVariable_2/Read/ReadVariableOp$dense_307/kernel/Read/ReadVariableOp"dense_307/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal_2/Read/ReadVariableOpcount_2/Read/ReadVariableOp#Adam/Variable/m/Read/ReadVariableOp%Adam/Variable/m_1/Read/ReadVariableOp%Adam/Variable/m_2/Read/ReadVariableOp+Adam/dense_307/kernel/m/Read/ReadVariableOp)Adam/dense_307/bias/m/Read/ReadVariableOp#Adam/Variable/v/Read/ReadVariableOp%Adam/Variable/v_1/Read/ReadVariableOp%Adam/Variable/v_2/Read/ReadVariableOp+Adam/dense_307/kernel/v/Read/ReadVariableOp)Adam/dense_307/bias/v/Read/ReadVariableOpConst*'
Tin 
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *)
f$R"
 __inference__traced_save_2864398
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameVariable
Variable_1
Variable_2dense_307/kerneldense_307/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1total_2count_2Adam/Variable/mAdam/Variable/m_1Adam/Variable/m_2Adam/dense_307/kernel/mAdam/dense_307/bias/mAdam/Variable/vAdam/Variable/v_1Adam/Variable/v_2Adam/dense_307/kernel/vAdam/dense_307/bias/v*&
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference__traced_restore_2864486??
?n
?
#__inference__traced_restore_2864486
file_prefix+
assignvariableop_variable:+
assignvariableop_1_variable_1:/
assignvariableop_2_variable_2:5
#assignvariableop_3_dense_307_kernel:/
!assignvariableop_4_dense_307_bias:&
assignvariableop_5_adam_iter:	 (
assignvariableop_6_adam_beta_1: (
assignvariableop_7_adam_beta_2: '
assignvariableop_8_adam_decay: /
%assignvariableop_9_adam_learning_rate: #
assignvariableop_10_total: #
assignvariableop_11_count: %
assignvariableop_12_total_1: %
assignvariableop_13_count_1: %
assignvariableop_14_total_2: %
assignvariableop_15_count_2: 5
#assignvariableop_16_adam_variable_m:3
%assignvariableop_17_adam_variable_m_1:7
%assignvariableop_18_adam_variable_m_2:=
+assignvariableop_19_adam_dense_307_kernel_m:7
)assignvariableop_20_adam_dense_307_bias_m:5
#assignvariableop_21_adam_variable_v:3
%assignvariableop_22_adam_variable_v_1:7
%assignvariableop_23_adam_variable_v_2:=
+assignvariableop_24_adam_dense_307_kernel_v:7
)assignvariableop_25_adam_dense_307_bias_v:
identity_27??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B1layer_with_weights-0/w/.ATTRIBUTES/VARIABLE_VALUEB1layer_with_weights-0/b/.ATTRIBUTES/VARIABLE_VALUEB1layer_with_weights-0/a/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEBMlayer_with_weights-0/w/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMlayer_with_weights-0/b/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMlayer_with_weights-0/a/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMlayer_with_weights-0/w/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMlayer_with_weights-0/b/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMlayer_with_weights-0/a/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*I
value@B>B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapesn
l:::::::::::::::::::::::::::*)
dtypes
2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOpassignvariableop_variableIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOpassignvariableop_1_variable_1Identity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOpassignvariableop_2_variable_2Identity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp#assignvariableop_3_dense_307_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp!assignvariableop_4_dense_307_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_iterIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_beta_1Identity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_beta_2Identity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_decayIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp%assignvariableop_9_adam_learning_rateIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOpassignvariableop_10_totalIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOpassignvariableop_11_countIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOpassignvariableop_12_total_1Identity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOpassignvariableop_13_count_1Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOpassignvariableop_14_total_2Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOpassignvariableop_15_count_2Identity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp#assignvariableop_16_adam_variable_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp%assignvariableop_17_adam_variable_m_1Identity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp%assignvariableop_18_adam_variable_m_2Identity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp+assignvariableop_19_adam_dense_307_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp)assignvariableop_20_adam_dense_307_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp#assignvariableop_21_adam_variable_vIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp%assignvariableop_22_adam_variable_v_1Identity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp%assignvariableop_23_adam_variable_v_2Identity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp+assignvariableop_24_adam_dense_307_kernel_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp)assignvariableop_25_adam_dense_307_bias_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_259
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_26Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_26?
Identity_27IdentityIdentity_26:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_27"#
identity_27Identity_27:output:0*I
_input_shapes8
6: : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?	
?
F__inference_dense_307_layer_call_and_return_conditional_losses_2864297

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
0__inference_sequential_151_layer_call_fn_2864204

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_sequential_151_layer_call_and_return_conditional_losses_28640792
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
0__inference_sequential_151_layer_call_fn_2864107
sine_85_input
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallsine_85_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_sequential_151_layer_call_and_return_conditional_losses_28640792
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
'
_output_shapes
:?????????
'
_user_specified_namesine_85_input
?
?
K__inference_sequential_151_layer_call_and_return_conditional_losses_2864139
sine_85_input!
sine_85_2864126:
sine_85_2864128:!
sine_85_2864130:#
dense_307_2864133:
dense_307_2864135:
identity??!dense_307/StatefulPartitionedCall?sine_85/StatefulPartitionedCall?
sine_85/StatefulPartitionedCallStatefulPartitionedCallsine_85_inputsine_85_2864126sine_85_2864128sine_85_2864130*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_sine_85_layer_call_and_return_conditional_losses_28639862!
sine_85/StatefulPartitionedCall?
!dense_307/StatefulPartitionedCallStatefulPartitionedCall(sine_85/StatefulPartitionedCall:output:0dense_307_2864133dense_307_2864135*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense_307_layer_call_and_return_conditional_losses_28640042#
!dense_307/StatefulPartitionedCall?
IdentityIdentity*dense_307/StatefulPartitionedCall:output:0"^dense_307/StatefulPartitionedCall ^sine_85/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????: : : : : 2F
!dense_307/StatefulPartitionedCall!dense_307/StatefulPartitionedCall2B
sine_85/StatefulPartitionedCallsine_85/StatefulPartitionedCall:V R
'
_output_shapes
:?????????
'
_user_specified_namesine_85_input
?
?
D__inference_sine_85_layer_call_and_return_conditional_losses_2864278

inputs-
mul_readvariableop_resource:)
add_readvariableop_resource:)
readvariableop_resource:
identity??ReadVariableOp?add/ReadVariableOp?mul/ReadVariableOp]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:?????????2
Cast?
mul/ReadVariableOpReadVariableOpmul_readvariableop_resource*
_output_shapes

:*
dtype02
mul/ReadVariableOpi
mulMulCast:y:0mul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
mul?
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:*
dtype02
add/ReadVariableOpj
addAddV2mul:z:0add/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
addL
SinSinadd:z:0*
T0*'
_output_shapes
:?????????2
Sinx
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:*
dtype02
ReadVariableOph
mul_1MulReadVariableOp:value:0Sin:y:0*
T0*'
_output_shapes
:?????????2
mul_1?
IdentityIdentity	mul_1:z:0^ReadVariableOp^add/ReadVariableOp^mul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:?????????: : : 2 
ReadVariableOpReadVariableOp2(
add/ReadVariableOpadd/ReadVariableOp2(
mul/ReadVariableOpmul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
K__inference_sequential_151_layer_call_and_return_conditional_losses_2864225

inputs5
#sine_85_mul_readvariableop_resource:1
#sine_85_add_readvariableop_resource:1
sine_85_readvariableop_resource::
(dense_307_matmul_readvariableop_resource:7
)dense_307_biasadd_readvariableop_resource:
identity?? dense_307/BiasAdd/ReadVariableOp?dense_307/MatMul/ReadVariableOp?sine_85/ReadVariableOp?sine_85/add/ReadVariableOp?sine_85/mul/ReadVariableOpm
sine_85/CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:?????????2
sine_85/Cast?
sine_85/mul/ReadVariableOpReadVariableOp#sine_85_mul_readvariableop_resource*
_output_shapes

:*
dtype02
sine_85/mul/ReadVariableOp?
sine_85/mulMulsine_85/Cast:y:0"sine_85/mul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sine_85/mul?
sine_85/add/ReadVariableOpReadVariableOp#sine_85_add_readvariableop_resource*
_output_shapes
:*
dtype02
sine_85/add/ReadVariableOp?
sine_85/addAddV2sine_85/mul:z:0"sine_85/add/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sine_85/addd
sine_85/SinSinsine_85/add:z:0*
T0*'
_output_shapes
:?????????2
sine_85/Sin?
sine_85/ReadVariableOpReadVariableOpsine_85_readvariableop_resource*
_output_shapes

:*
dtype02
sine_85/ReadVariableOp?
sine_85/mul_1Mulsine_85/ReadVariableOp:value:0sine_85/Sin:y:0*
T0*'
_output_shapes
:?????????2
sine_85/mul_1?
dense_307/MatMul/ReadVariableOpReadVariableOp(dense_307_matmul_readvariableop_resource*
_output_shapes

:*
dtype02!
dense_307/MatMul/ReadVariableOp?
dense_307/MatMulMatMulsine_85/mul_1:z:0'dense_307/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_307/MatMul?
 dense_307/BiasAdd/ReadVariableOpReadVariableOp)dense_307_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 dense_307/BiasAdd/ReadVariableOp?
dense_307/BiasAddBiasAdddense_307/MatMul:product:0(dense_307/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_307/BiasAdd?
IdentityIdentitydense_307/BiasAdd:output:0!^dense_307/BiasAdd/ReadVariableOp ^dense_307/MatMul/ReadVariableOp^sine_85/ReadVariableOp^sine_85/add/ReadVariableOp^sine_85/mul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????: : : : : 2D
 dense_307/BiasAdd/ReadVariableOp dense_307/BiasAdd/ReadVariableOp2B
dense_307/MatMul/ReadVariableOpdense_307/MatMul/ReadVariableOp20
sine_85/ReadVariableOpsine_85/ReadVariableOp28
sine_85/add/ReadVariableOpsine_85/add/ReadVariableOp28
sine_85/mul/ReadVariableOpsine_85/mul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
K__inference_sequential_151_layer_call_and_return_conditional_losses_2864011

inputs!
sine_85_2863987:
sine_85_2863989:!
sine_85_2863991:#
dense_307_2864005:
dense_307_2864007:
identity??!dense_307/StatefulPartitionedCall?sine_85/StatefulPartitionedCall?
sine_85/StatefulPartitionedCallStatefulPartitionedCallinputssine_85_2863987sine_85_2863989sine_85_2863991*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_sine_85_layer_call_and_return_conditional_losses_28639862!
sine_85/StatefulPartitionedCall?
!dense_307/StatefulPartitionedCallStatefulPartitionedCall(sine_85/StatefulPartitionedCall:output:0dense_307_2864005dense_307_2864007*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense_307_layer_call_and_return_conditional_losses_28640042#
!dense_307/StatefulPartitionedCall?
IdentityIdentity*dense_307/StatefulPartitionedCall:output:0"^dense_307/StatefulPartitionedCall ^sine_85/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????: : : : : 2F
!dense_307/StatefulPartitionedCall!dense_307/StatefulPartitionedCall2B
sine_85/StatefulPartitionedCallsine_85/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
F__inference_dense_307_layer_call_and_return_conditional_losses_2864004

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
K__inference_sequential_151_layer_call_and_return_conditional_losses_2864123
sine_85_input!
sine_85_2864110:
sine_85_2864112:!
sine_85_2864114:#
dense_307_2864117:
dense_307_2864119:
identity??!dense_307/StatefulPartitionedCall?sine_85/StatefulPartitionedCall?
sine_85/StatefulPartitionedCallStatefulPartitionedCallsine_85_inputsine_85_2864110sine_85_2864112sine_85_2864114*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_sine_85_layer_call_and_return_conditional_losses_28639862!
sine_85/StatefulPartitionedCall?
!dense_307/StatefulPartitionedCallStatefulPartitionedCall(sine_85/StatefulPartitionedCall:output:0dense_307_2864117dense_307_2864119*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense_307_layer_call_and_return_conditional_losses_28640042#
!dense_307/StatefulPartitionedCall?
IdentityIdentity*dense_307/StatefulPartitionedCall:output:0"^dense_307/StatefulPartitionedCall ^sine_85/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????: : : : : 2F
!dense_307/StatefulPartitionedCall!dense_307/StatefulPartitionedCall2B
sine_85/StatefulPartitionedCallsine_85/StatefulPartitionedCall:V R
'
_output_shapes
:?????????
'
_user_specified_namesine_85_input
?
?
D__inference_sine_85_layer_call_and_return_conditional_losses_2863986

inputs-
mul_readvariableop_resource:)
add_readvariableop_resource:)
readvariableop_resource:
identity??ReadVariableOp?add/ReadVariableOp?mul/ReadVariableOp]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:?????????2
Cast?
mul/ReadVariableOpReadVariableOpmul_readvariableop_resource*
_output_shapes

:*
dtype02
mul/ReadVariableOpi
mulMulCast:y:0mul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
mul?
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:*
dtype02
add/ReadVariableOpj
addAddV2mul:z:0add/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
addL
SinSinadd:z:0*
T0*'
_output_shapes
:?????????2
Sinx
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:*
dtype02
ReadVariableOph
mul_1MulReadVariableOp:value:0Sin:y:0*
T0*'
_output_shapes
:?????????2
mul_1?
IdentityIdentity	mul_1:z:0^ReadVariableOp^add/ReadVariableOp^mul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:?????????: : : 2 
ReadVariableOpReadVariableOp2(
add/ReadVariableOpadd/ReadVariableOp2(
mul/ReadVariableOpmul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
%__inference_signature_wrapper_2864174
sine_85_input
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallsine_85_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8? *+
f&R$
"__inference__wrapped_model_28639642
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
'
_output_shapes
:?????????
'
_user_specified_namesine_85_input
?
?
0__inference_sequential_151_layer_call_fn_2864189

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_sequential_151_layer_call_and_return_conditional_losses_28640112
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
)__inference_sine_85_layer_call_fn_2864263

inputs
unknown:
	unknown_0:
	unknown_1:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_sine_85_layer_call_and_return_conditional_losses_28639862
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:?????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
0__inference_sequential_151_layer_call_fn_2864024
sine_85_input
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallsine_85_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_sequential_151_layer_call_and_return_conditional_losses_28640112
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
'
_output_shapes
:?????????
'
_user_specified_namesine_85_input
?9
?

 __inference__traced_save_2864398
file_prefix'
#savev2_variable_read_readvariableop)
%savev2_variable_1_read_readvariableop)
%savev2_variable_2_read_readvariableop/
+savev2_dense_307_kernel_read_readvariableop-
)savev2_dense_307_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop&
"savev2_total_2_read_readvariableop&
"savev2_count_2_read_readvariableop.
*savev2_adam_variable_m_read_readvariableop0
,savev2_adam_variable_m_1_read_readvariableop0
,savev2_adam_variable_m_2_read_readvariableop6
2savev2_adam_dense_307_kernel_m_read_readvariableop4
0savev2_adam_dense_307_bias_m_read_readvariableop.
*savev2_adam_variable_v_read_readvariableop0
,savev2_adam_variable_v_1_read_readvariableop0
,savev2_adam_variable_v_2_read_readvariableop6
2savev2_adam_dense_307_kernel_v_read_readvariableop4
0savev2_adam_dense_307_bias_v_read_readvariableop
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
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B1layer_with_weights-0/w/.ATTRIBUTES/VARIABLE_VALUEB1layer_with_weights-0/b/.ATTRIBUTES/VARIABLE_VALUEB1layer_with_weights-0/a/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEBMlayer_with_weights-0/w/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMlayer_with_weights-0/b/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMlayer_with_weights-0/a/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMlayer_with_weights-0/w/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMlayer_with_weights-0/b/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMlayer_with_weights-0/a/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*I
value@B>B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?

SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0#savev2_variable_read_readvariableop%savev2_variable_1_read_readvariableop%savev2_variable_2_read_readvariableop+savev2_dense_307_kernel_read_readvariableop)savev2_dense_307_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop"savev2_total_2_read_readvariableop"savev2_count_2_read_readvariableop*savev2_adam_variable_m_read_readvariableop,savev2_adam_variable_m_1_read_readvariableop,savev2_adam_variable_m_2_read_readvariableop2savev2_adam_dense_307_kernel_m_read_readvariableop0savev2_adam_dense_307_bias_m_read_readvariableop*savev2_adam_variable_v_read_readvariableop,savev2_adam_variable_v_1_read_readvariableop,savev2_adam_variable_v_2_read_readvariableop2savev2_adam_dense_307_kernel_v_read_readvariableop0savev2_adam_dense_307_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *)
dtypes
2	2
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

identity_1Identity_1:output:0*?
_input_shapes?
?: :::::: : : : : : : : : : : ::::::::::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

::$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

::$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: 
?
?
+__inference_dense_307_layer_call_fn_2864287

inputs
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense_307_layer_call_and_return_conditional_losses_28640042
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
"__inference__wrapped_model_2863964
sine_85_inputD
2sequential_151_sine_85_mul_readvariableop_resource:@
2sequential_151_sine_85_add_readvariableop_resource:@
.sequential_151_sine_85_readvariableop_resource:I
7sequential_151_dense_307_matmul_readvariableop_resource:F
8sequential_151_dense_307_biasadd_readvariableop_resource:
identity??/sequential_151/dense_307/BiasAdd/ReadVariableOp?.sequential_151/dense_307/MatMul/ReadVariableOp?%sequential_151/sine_85/ReadVariableOp?)sequential_151/sine_85/add/ReadVariableOp?)sequential_151/sine_85/mul/ReadVariableOp?
sequential_151/sine_85/CastCastsine_85_input*

DstT0*

SrcT0*'
_output_shapes
:?????????2
sequential_151/sine_85/Cast?
)sequential_151/sine_85/mul/ReadVariableOpReadVariableOp2sequential_151_sine_85_mul_readvariableop_resource*
_output_shapes

:*
dtype02+
)sequential_151/sine_85/mul/ReadVariableOp?
sequential_151/sine_85/mulMulsequential_151/sine_85/Cast:y:01sequential_151/sine_85/mul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential_151/sine_85/mul?
)sequential_151/sine_85/add/ReadVariableOpReadVariableOp2sequential_151_sine_85_add_readvariableop_resource*
_output_shapes
:*
dtype02+
)sequential_151/sine_85/add/ReadVariableOp?
sequential_151/sine_85/addAddV2sequential_151/sine_85/mul:z:01sequential_151/sine_85/add/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential_151/sine_85/add?
sequential_151/sine_85/SinSinsequential_151/sine_85/add:z:0*
T0*'
_output_shapes
:?????????2
sequential_151/sine_85/Sin?
%sequential_151/sine_85/ReadVariableOpReadVariableOp.sequential_151_sine_85_readvariableop_resource*
_output_shapes

:*
dtype02'
%sequential_151/sine_85/ReadVariableOp?
sequential_151/sine_85/mul_1Mul-sequential_151/sine_85/ReadVariableOp:value:0sequential_151/sine_85/Sin:y:0*
T0*'
_output_shapes
:?????????2
sequential_151/sine_85/mul_1?
.sequential_151/dense_307/MatMul/ReadVariableOpReadVariableOp7sequential_151_dense_307_matmul_readvariableop_resource*
_output_shapes

:*
dtype020
.sequential_151/dense_307/MatMul/ReadVariableOp?
sequential_151/dense_307/MatMulMatMul sequential_151/sine_85/mul_1:z:06sequential_151/dense_307/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2!
sequential_151/dense_307/MatMul?
/sequential_151/dense_307/BiasAdd/ReadVariableOpReadVariableOp8sequential_151_dense_307_biasadd_readvariableop_resource*
_output_shapes
:*
dtype021
/sequential_151/dense_307/BiasAdd/ReadVariableOp?
 sequential_151/dense_307/BiasAddBiasAdd)sequential_151/dense_307/MatMul:product:07sequential_151/dense_307/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2"
 sequential_151/dense_307/BiasAdd?
IdentityIdentity)sequential_151/dense_307/BiasAdd:output:00^sequential_151/dense_307/BiasAdd/ReadVariableOp/^sequential_151/dense_307/MatMul/ReadVariableOp&^sequential_151/sine_85/ReadVariableOp*^sequential_151/sine_85/add/ReadVariableOp*^sequential_151/sine_85/mul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????: : : : : 2b
/sequential_151/dense_307/BiasAdd/ReadVariableOp/sequential_151/dense_307/BiasAdd/ReadVariableOp2`
.sequential_151/dense_307/MatMul/ReadVariableOp.sequential_151/dense_307/MatMul/ReadVariableOp2N
%sequential_151/sine_85/ReadVariableOp%sequential_151/sine_85/ReadVariableOp2V
)sequential_151/sine_85/add/ReadVariableOp)sequential_151/sine_85/add/ReadVariableOp2V
)sequential_151/sine_85/mul/ReadVariableOp)sequential_151/sine_85/mul/ReadVariableOp:V R
'
_output_shapes
:?????????
'
_user_specified_namesine_85_input
?
?
K__inference_sequential_151_layer_call_and_return_conditional_losses_2864079

inputs!
sine_85_2864066:
sine_85_2864068:!
sine_85_2864070:#
dense_307_2864073:
dense_307_2864075:
identity??!dense_307/StatefulPartitionedCall?sine_85/StatefulPartitionedCall?
sine_85/StatefulPartitionedCallStatefulPartitionedCallinputssine_85_2864066sine_85_2864068sine_85_2864070*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_sine_85_layer_call_and_return_conditional_losses_28639862!
sine_85/StatefulPartitionedCall?
!dense_307/StatefulPartitionedCallStatefulPartitionedCall(sine_85/StatefulPartitionedCall:output:0dense_307_2864073dense_307_2864075*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense_307_layer_call_and_return_conditional_losses_28640042#
!dense_307/StatefulPartitionedCall?
IdentityIdentity*dense_307/StatefulPartitionedCall:output:0"^dense_307/StatefulPartitionedCall ^sine_85/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????: : : : : 2F
!dense_307/StatefulPartitionedCall!dense_307/StatefulPartitionedCall2B
sine_85/StatefulPartitionedCallsine_85/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
K__inference_sequential_151_layer_call_and_return_conditional_losses_2864246

inputs5
#sine_85_mul_readvariableop_resource:1
#sine_85_add_readvariableop_resource:1
sine_85_readvariableop_resource::
(dense_307_matmul_readvariableop_resource:7
)dense_307_biasadd_readvariableop_resource:
identity?? dense_307/BiasAdd/ReadVariableOp?dense_307/MatMul/ReadVariableOp?sine_85/ReadVariableOp?sine_85/add/ReadVariableOp?sine_85/mul/ReadVariableOpm
sine_85/CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:?????????2
sine_85/Cast?
sine_85/mul/ReadVariableOpReadVariableOp#sine_85_mul_readvariableop_resource*
_output_shapes

:*
dtype02
sine_85/mul/ReadVariableOp?
sine_85/mulMulsine_85/Cast:y:0"sine_85/mul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sine_85/mul?
sine_85/add/ReadVariableOpReadVariableOp#sine_85_add_readvariableop_resource*
_output_shapes
:*
dtype02
sine_85/add/ReadVariableOp?
sine_85/addAddV2sine_85/mul:z:0"sine_85/add/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sine_85/addd
sine_85/SinSinsine_85/add:z:0*
T0*'
_output_shapes
:?????????2
sine_85/Sin?
sine_85/ReadVariableOpReadVariableOpsine_85_readvariableop_resource*
_output_shapes

:*
dtype02
sine_85/ReadVariableOp?
sine_85/mul_1Mulsine_85/ReadVariableOp:value:0sine_85/Sin:y:0*
T0*'
_output_shapes
:?????????2
sine_85/mul_1?
dense_307/MatMul/ReadVariableOpReadVariableOp(dense_307_matmul_readvariableop_resource*
_output_shapes

:*
dtype02!
dense_307/MatMul/ReadVariableOp?
dense_307/MatMulMatMulsine_85/mul_1:z:0'dense_307/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_307/MatMul?
 dense_307/BiasAdd/ReadVariableOpReadVariableOp)dense_307_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 dense_307/BiasAdd/ReadVariableOp?
dense_307/BiasAddBiasAdddense_307/MatMul:product:0(dense_307/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_307/BiasAdd?
IdentityIdentitydense_307/BiasAdd:output:0!^dense_307/BiasAdd/ReadVariableOp ^dense_307/MatMul/ReadVariableOp^sine_85/ReadVariableOp^sine_85/add/ReadVariableOp^sine_85/mul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????: : : : : 2D
 dense_307/BiasAdd/ReadVariableOp dense_307/BiasAdd/ReadVariableOp2B
dense_307/MatMul/ReadVariableOpdense_307/MatMul/ReadVariableOp20
sine_85/ReadVariableOpsine_85/ReadVariableOp28
sine_85/add/ReadVariableOpsine_85/add/ReadVariableOp28
sine_85/mul/ReadVariableOpsine_85/mul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
G
sine_85_input6
serving_default_sine_85_input:0?????????=
	dense_3070
StatefulPartitionedCall:0?????????tensorflow/serving/predict:?i
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
	optimizer
trainable_variables
regularization_losses
	variables
	keras_api

signatures
E_default_save_signature
F__call__
*G&call_and_return_all_conditional_losses"?
_tf_keras_sequential?{"name": "sequential_151", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Sequential", "config": {"name": "sequential_151", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "int32", "sparse": false, "ragged": false, "name": "sine_85_input"}}, {"class_name": "Sine", "config": {"name": "sine_85", "trainable": true, "dtype": "float32", "w": [[0.017453264445066452]], "b": [-0.0003092127153649926], "a": [[0.6390385627746582]]}}, {"class_name": "Dense", "config": {"name": "dense_307", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "shared_object_id": 5, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1]}, "int32", "sine_85_input"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_151", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "int32", "sparse": false, "ragged": false, "name": "sine_85_input"}, "shared_object_id": 0}, {"class_name": "Sine", "config": {"name": "sine_85", "trainable": true, "dtype": "float32", "w": [[0.017453264445066452]], "b": [-0.0003092127153649926], "a": [[0.6390385627746582]]}, "shared_object_id": 1}, {"class_name": "Dense", "config": {"name": "dense_307", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 2}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 3}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 4}]}}, "training_config": {"loss": "mean_squared_error", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "mae", "dtype": "float32", "fn": "mean_absolute_error"}, "shared_object_id": 6}, {"class_name": "MeanMetricWrapper", "config": {"name": "mse", "dtype": "float32", "fn": "mean_squared_error"}, "shared_object_id": 7}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?
	w

b
a
trainable_variables
regularization_losses
	variables
	keras_api
H__call__
*I&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "sine_85", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Sine", "config": {"name": "sine_85", "trainable": true, "dtype": "float32", "w": [[0.017453264445066452]], "b": [-0.0003092127153649926], "a": [[0.6390385627746582]]}, "shared_object_id": 1}
?

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
J__call__
*K&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "dense_307", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_307", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 2}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 3}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 4, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1}}, "shared_object_id": 8}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1]}}
?
iter

beta_1

beta_2
	decay
learning_rate	m;
m<m=m>m?	v@
vAvBvCvD"
	optimizer
C
	0

1
2
3
4"
trackable_list_wrapper
 "
trackable_list_wrapper
C
	0

1
2
3
4"
trackable_list_wrapper
?

layers
layer_regularization_losses
layer_metrics
trainable_variables
non_trainable_variables
regularization_losses
metrics
	variables
F__call__
E_default_save_signature
*G&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses"
_generic_user_object
,
Lserving_default"
signature_map
:2Variable
:2Variable
:2Variable
5
	0

1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
5
	0

1
2"
trackable_list_wrapper
?

 layers
!layer_regularization_losses
"layer_metrics
trainable_variables
#non_trainable_variables
regularization_losses
$metrics
	variables
H__call__
*I&call_and_return_all_conditional_losses
&I"call_and_return_conditional_losses"
_generic_user_object
": 2dense_307/kernel
:2dense_307/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?

%layers
&layer_regularization_losses
'layer_metrics
trainable_variables
(non_trainable_variables
regularization_losses
)metrics
	variables
J__call__
*K&call_and_return_all_conditional_losses
&K"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
5
*0
+1
,2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
	-total
	.count
/	variables
0	keras_api"?
_tf_keras_metric?{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}, "shared_object_id": 9}
?
	1total
	2count
3
_fn_kwargs
4	variables
5	keras_api"?
_tf_keras_metric?{"class_name": "MeanMetricWrapper", "name": "mae", "dtype": "float32", "config": {"name": "mae", "dtype": "float32", "fn": "mean_absolute_error"}, "shared_object_id": 6}
?
	6total
	7count
8
_fn_kwargs
9	variables
:	keras_api"?
_tf_keras_metric?{"class_name": "MeanMetricWrapper", "name": "mse", "dtype": "float32", "config": {"name": "mse", "dtype": "float32", "fn": "mean_squared_error"}, "shared_object_id": 7}
:  (2total
:  (2count
.
-0
.1"
trackable_list_wrapper
-
/	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
10
21"
trackable_list_wrapper
-
4	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
60
71"
trackable_list_wrapper
-
9	variables"
_generic_user_object
:2Adam/Variable/m
:2Adam/Variable/m
:2Adam/Variable/m
':%2Adam/dense_307/kernel/m
!:2Adam/dense_307/bias/m
:2Adam/Variable/v
:2Adam/Variable/v
:2Adam/Variable/v
':%2Adam/dense_307/kernel/v
!:2Adam/dense_307/bias/v
?2?
"__inference__wrapped_model_2863964?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *,?)
'?$
sine_85_input?????????
?2?
0__inference_sequential_151_layer_call_fn_2864024
0__inference_sequential_151_layer_call_fn_2864189
0__inference_sequential_151_layer_call_fn_2864204
0__inference_sequential_151_layer_call_fn_2864107?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
K__inference_sequential_151_layer_call_and_return_conditional_losses_2864225
K__inference_sequential_151_layer_call_and_return_conditional_losses_2864246
K__inference_sequential_151_layer_call_and_return_conditional_losses_2864123
K__inference_sequential_151_layer_call_and_return_conditional_losses_2864139?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
)__inference_sine_85_layer_call_fn_2864263?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_sine_85_layer_call_and_return_conditional_losses_2864278?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_dense_307_layer_call_fn_2864287?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_dense_307_layer_call_and_return_conditional_losses_2864297?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
%__inference_signature_wrapper_2864174sine_85_input"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
"__inference__wrapped_model_2863964v	
6?3
,?)
'?$
sine_85_input?????????
? "5?2
0
	dense_307#? 
	dense_307??????????
F__inference_dense_307_layer_call_and_return_conditional_losses_2864297\/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? ~
+__inference_dense_307_layer_call_fn_2864287O/?,
%?"
 ?
inputs?????????
? "???????????
K__inference_sequential_151_layer_call_and_return_conditional_losses_2864123n	
>?;
4?1
'?$
sine_85_input?????????
p 

 
? "%?"
?
0?????????
? ?
K__inference_sequential_151_layer_call_and_return_conditional_losses_2864139n	
>?;
4?1
'?$
sine_85_input?????????
p

 
? "%?"
?
0?????????
? ?
K__inference_sequential_151_layer_call_and_return_conditional_losses_2864225g	
7?4
-?*
 ?
inputs?????????
p 

 
? "%?"
?
0?????????
? ?
K__inference_sequential_151_layer_call_and_return_conditional_losses_2864246g	
7?4
-?*
 ?
inputs?????????
p

 
? "%?"
?
0?????????
? ?
0__inference_sequential_151_layer_call_fn_2864024a	
>?;
4?1
'?$
sine_85_input?????????
p 

 
? "???????????
0__inference_sequential_151_layer_call_fn_2864107a	
>?;
4?1
'?$
sine_85_input?????????
p

 
? "???????????
0__inference_sequential_151_layer_call_fn_2864189Z	
7?4
-?*
 ?
inputs?????????
p 

 
? "???????????
0__inference_sequential_151_layer_call_fn_2864204Z	
7?4
-?*
 ?
inputs?????????
p

 
? "???????????
%__inference_signature_wrapper_2864174?	
G?D
? 
=?:
8
sine_85_input'?$
sine_85_input?????????"5?2
0
	dense_307#? 
	dense_307??????????
D__inference_sine_85_layer_call_and_return_conditional_losses_2864278]	
/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? }
)__inference_sine_85_layer_call_fn_2864263P	
/?,
%?"
 ?
inputs?????????
? "??????????