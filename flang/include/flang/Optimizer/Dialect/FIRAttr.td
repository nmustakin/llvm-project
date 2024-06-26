//===- FIRAttr.td - FIR Attributes -------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the FIR dialect attributes.
//
//===----------------------------------------------------------------------===//

#ifndef FIR_DIALECT_FIR_ATTRS
#define FIR_DIALECT_FIR_ATTRS

include "flang/Optimizer/Dialect/FIRDialect.td"
include "mlir/IR/EnumAttr.td"

class fir_Attr<string name> : AttrDef<FIROpsDialect, name>;

def FIRnoAttributes  : I32BitEnumAttrCaseNone<"None">;
def FIRallocatable  : I32BitEnumAttrCaseBit<"allocatable", 0>;
def FIRasynchronous : I32BitEnumAttrCaseBit<"asynchronous", 1>;
def FIRbind_c       : I32BitEnumAttrCaseBit<"bind_c", 2>;
def FIRcontiguous   : I32BitEnumAttrCaseBit<"contiguous", 3>;
def FIRintent_in    : I32BitEnumAttrCaseBit<"intent_in", 4>;
def FIRintent_inout : I32BitEnumAttrCaseBit<"intent_inout", 5>;
def FIRintent_out   : I32BitEnumAttrCaseBit<"intent_out", 6>;
def FIRoptional     : I32BitEnumAttrCaseBit<"optional", 7>;
def FIRparameter    : I32BitEnumAttrCaseBit<"parameter", 8>;
def FIRpointer      : I32BitEnumAttrCaseBit<"pointer", 9>;
def FIRtarget       : I32BitEnumAttrCaseBit<"target", 10>;
def FIRvalue        : I32BitEnumAttrCaseBit<"value", 11>;
def FIRvolatile     : I32BitEnumAttrCaseBit<"fortran_volatile", 12, "volatile">;
def FIRHostAssoc    : I32BitEnumAttrCaseBit<"host_assoc", 13>;

def fir_FortranVariableFlagsEnum : I32BitEnumAttr<
    "FortranVariableFlagsEnum",
    "Fortran variable attributes",
    [FIRnoAttributes, FIRallocatable, FIRasynchronous, FIRbind_c, FIRcontiguous,
     FIRintent_in, FIRintent_inout, FIRintent_out, FIRoptional, FIRparameter,
     FIRpointer, FIRtarget, FIRvalue, FIRvolatile, FIRHostAssoc]> {
  let separator = ", ";
  let cppNamespace = "::fir";
  let printBitEnumPrimaryGroups = 1;
}

def fir_FortranVariableFlagsAttr : fir_Attr<"FortranVariableFlags"> {
  let mnemonic = "var_attrs";

  let parameters = (ins
    "FortranVariableFlagsEnum":$flags
  );
  let hasCustomAssemblyFormat = 1;
  let returnType = "::fir::FortranVariableFlagsEnum";
  let convertFromStorage = "$_self.getFlags()";
  let constBuilderCall =
        "::fir::FortranVariableFlagsAttr::get($_builder.getContext(), $0)";
}

def fir_BoxFieldAttr : I32EnumAttr<
    "BoxFieldAttr", "",
    [
      I32EnumAttrCase<"base_addr", 0>,
      I32EnumAttrCase<"derived_type", 1>
    ]> {
  let cppNamespace = "fir";
}

// mlir::SideEffects::Resource for modelling operations which add debugging information
def DebuggingResource : Resource<"::fir::DebuggingResource">;

//===----------------------------------------------------------------------===//
// CUDA Fortran specific attributes
//===----------------------------------------------------------------------===//

def fir_CUDADataAttribute : I32EnumAttr<
    "CUDADataAttribute",
    "CUDA Fortran variable attributes",
    [
      I32EnumAttrCase<"Constant", 0, "constant">,
      I32EnumAttrCase<"Device", 1, "device">,
      I32EnumAttrCase<"Managed", 2, "managed">,
      I32EnumAttrCase<"Pinned", 3, "pinned">,
      I32EnumAttrCase<"Shared", 4, "shared">,
      I32EnumAttrCase<"Unified", 5, "unified">,
      // Texture is omitted since it is obsolete and rejected by semantic.
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::fir";
}

def fir_CUDADataAttributeAttr :
    EnumAttr<FIROpsDialect, fir_CUDADataAttribute, "cuda"> {
  let assemblyFormat = [{ ```<` $value `>` }];
}

def fir_CUDAProcAttribute : I32EnumAttr<
    "CUDAProcAttribute", "CUDA Fortran procedure attributes",
    [
      I32EnumAttrCase<"Host", 0, "host">,
      I32EnumAttrCase<"Device", 1, "device">,
      I32EnumAttrCase<"HostDevice", 2, "host_device">,
      I32EnumAttrCase<"Global", 3, "global">,
      I32EnumAttrCase<"GridGlobal", 4, "grid_global">,
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::fir";
}

def fir_CUDAProcAttributeAttr :
    EnumAttr<FIROpsDialect, fir_CUDAProcAttribute, "cuda_proc"> {
  let assemblyFormat = [{ ```<` $value `>` }];
}

def fir_CUDALaunchBoundsAttr : fir_Attr<"CUDALaunchBounds"> {
  let mnemonic = "launch_bounds";

  let parameters = (ins
    "mlir::IntegerAttr":$maxTPB,
    "mlir::IntegerAttr":$minBPM,
    OptionalParameter<"mlir::IntegerAttr">:$upperBoundClusterSize
  );

  let assemblyFormat = "`<` struct(params) `>`";
}

def fir_CUDAClusterDimsAttr : fir_Attr<"CUDAClusterDims"> {
  let mnemonic = "cluster_dims";

  let parameters = (ins
    "mlir::IntegerAttr":$x,
    "mlir::IntegerAttr":$y,
    "mlir::IntegerAttr":$z
  );

  let assemblyFormat = "`<` struct(params) `>`";
}

def fir_CUDADataTransferKind : I32EnumAttr<
    "CUDADataTransferKind", "CUDA Fortran data transfer kind",
    [
      I32EnumAttrCase<"DeviceHost", 0, "device_host">,
      I32EnumAttrCase<"HostDevice", 1, "host_device">,
      I32EnumAttrCase<"DeviceDevice", 2, "device_device">,
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::fir";
}

def fir_CUDADataTransferKindAttr :
    EnumAttr<FIROpsDialect, fir_CUDADataTransferKind, "cuda_transfer"> {
  let assemblyFormat = [{ ```<` $value `>` }];
}

#endif // FIR_DIALECT_FIR_ATTRS
