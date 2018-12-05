# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: caffeine/proto/layer_param.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='caffeine/proto/layer_param.proto',
  package='caffeine',
  serialized_pb=_b('\n caffeine/proto/layer_param.proto\x12\x08\x63\x61\x66\x66\x65ine\",\n\x0eLayerParameter\x12\x0c\n\x04name\x18\x01 \x02(\t\x12\x0c\n\x04type\x18\x02 \x02(\t\"r\n\x0f\x46illerParameter\x12\x0c\n\x04type\x18\x01 \x02(\t\x12\x10\n\x05value\x18\x02 \x01(\x02:\x01\x30\x12\x0e\n\x03min\x18\x03 \x01(\x02:\x01\x30\x12\x0e\n\x03max\x18\x04 \x01(\x02:\x01\x31\x12\x0f\n\x04mean\x18\x05 \x01(\x02:\x01\x30\x12\x0e\n\x03std\x18\x06 \x01(\x02:\x01\x31\"q\n\tBlobProto\x12\x0e\n\x03num\x18\x01 \x01(\x05:\x01\x30\x12\x13\n\x08\x63hannels\x18\x02 \x01(\x05:\x01\x30\x12\x11\n\x06height\x18\x03 \x01(\x05:\x01\x30\x12\x10\n\x05width\x18\x04 \x01(\x05:\x01\x30\x12\x0c\n\x04\x64\x61ta\x18\x05 \x03(\x02\x12\x0c\n\x04\x64iff\x18\x06 \x03(\x02')
)
_sym_db.RegisterFileDescriptor(DESCRIPTOR)




_LAYERPARAMETER = _descriptor.Descriptor(
  name='LayerParameter',
  full_name='caffeine.LayerParameter',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='name', full_name='caffeine.LayerParameter.name', index=0,
      number=1, type=9, cpp_type=9, label=2,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='type', full_name='caffeine.LayerParameter.type', index=1,
      number=2, type=9, cpp_type=9, label=2,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=46,
  serialized_end=90,
)


_FILLERPARAMETER = _descriptor.Descriptor(
  name='FillerParameter',
  full_name='caffeine.FillerParameter',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='type', full_name='caffeine.FillerParameter.type', index=0,
      number=1, type=9, cpp_type=9, label=2,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='value', full_name='caffeine.FillerParameter.value', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='min', full_name='caffeine.FillerParameter.min', index=2,
      number=3, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='max', full_name='caffeine.FillerParameter.max', index=3,
      number=4, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=1,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='mean', full_name='caffeine.FillerParameter.mean', index=4,
      number=5, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='std', full_name='caffeine.FillerParameter.std', index=5,
      number=6, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=1,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=92,
  serialized_end=206,
)


_BLOBPROTO = _descriptor.Descriptor(
  name='BlobProto',
  full_name='caffeine.BlobProto',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='num', full_name='caffeine.BlobProto.num', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='channels', full_name='caffeine.BlobProto.channels', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='height', full_name='caffeine.BlobProto.height', index=2,
      number=3, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='width', full_name='caffeine.BlobProto.width', index=3,
      number=4, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='data', full_name='caffeine.BlobProto.data', index=4,
      number=5, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='diff', full_name='caffeine.BlobProto.diff', index=5,
      number=6, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=208,
  serialized_end=321,
)

DESCRIPTOR.message_types_by_name['LayerParameter'] = _LAYERPARAMETER
DESCRIPTOR.message_types_by_name['FillerParameter'] = _FILLERPARAMETER
DESCRIPTOR.message_types_by_name['BlobProto'] = _BLOBPROTO

LayerParameter = _reflection.GeneratedProtocolMessageType('LayerParameter', (_message.Message,), dict(
  DESCRIPTOR = _LAYERPARAMETER,
  __module__ = 'caffeine.proto.layer_param_pb2'
  # @@protoc_insertion_point(class_scope:caffeine.LayerParameter)
  ))
_sym_db.RegisterMessage(LayerParameter)

FillerParameter = _reflection.GeneratedProtocolMessageType('FillerParameter', (_message.Message,), dict(
  DESCRIPTOR = _FILLERPARAMETER,
  __module__ = 'caffeine.proto.layer_param_pb2'
  # @@protoc_insertion_point(class_scope:caffeine.FillerParameter)
  ))
_sym_db.RegisterMessage(FillerParameter)

BlobProto = _reflection.GeneratedProtocolMessageType('BlobProto', (_message.Message,), dict(
  DESCRIPTOR = _BLOBPROTO,
  __module__ = 'caffeine.proto.layer_param_pb2'
  # @@protoc_insertion_point(class_scope:caffeine.BlobProto)
  ))
_sym_db.RegisterMessage(BlobProto)


# @@protoc_insertion_point(module_scope)
