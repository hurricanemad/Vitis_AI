# SPDX-License-Identifier: Apache-2.0

# automatically generated by the FlatBuffers compiler, do not modify

# namespace: tflite

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class DepthToSpaceOptions(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = DepthToSpaceOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsDepthToSpaceOptions(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)
    @classmethod
    def DepthToSpaceOptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed)

    # DepthToSpaceOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # DepthToSpaceOptions
    def BlockSize(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

def Start(builder): builder.StartObject(1)
def DepthToSpaceOptionsStart(builder):
    """This method is deprecated. Please switch to Start."""
    return Start(builder)
def AddBlockSize(builder, blockSize): builder.PrependInt32Slot(0, blockSize, 0)
def DepthToSpaceOptionsAddBlockSize(builder, blockSize):
    """This method is deprecated. Please switch to AddBlockSize."""
    return AddBlockSize(builder, blockSize)
def End(builder): return builder.EndObject()
def DepthToSpaceOptionsEnd(builder):
    """This method is deprecated. Please switch to End."""
    return End(builder)