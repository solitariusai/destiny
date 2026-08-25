use pyo3::prelude::*;
use pyo3::exceptions::PyTypeError;
use numpy::{
    PyUntypedArrayMethods, 
    PyArrayDescrMethods, 
    PyUntypedArray, 
    PyArrayDescr
};
use safetensors::tensor::{TensorView, Dtype};

pub fn extract_dtype(dtype: Bound<'_, PyArrayDescr>) -> PyResult<Dtype>{
    let name: String = dtype.typeobj().name()?.to_str()?.to_owned();
    match name.as_str() {
        "float4_e2m1fn"     => Ok(Dtype::F4),
        "bool_" | "bool"    => Ok(Dtype::BOOL),
        "uint8"             => Ok(Dtype::U8),
        "int8"              => Ok(Dtype::I8),
        "float8_e5m2"       => Ok(Dtype::F8_E5M2),
        "float8_e4m3fn"     => Ok(Dtype::F8_E4M3),
        "float8_e8m0fnu"    => Ok(Dtype::F8_E8M0),
        "float8_e4m3fnuz"   => Ok(Dtype::F8_E4M3FNUZ),
        "float8_e5m2fnuz"   => Ok(Dtype::F8_E5M2FNUZ),
        "int16"             => Ok(Dtype::I16),
        "uint16"            => Ok(Dtype::U16),
        "int32"             => Ok(Dtype::I32),
        "uint32"            => Ok(Dtype::U32),
        "int64"             => Ok(Dtype::I64),
        "uint64"            => Ok(Dtype::U64),
        "float16"           => Ok(Dtype::F16),
        "bfloat16"          => Ok(Dtype::BF16),
        "float32"           => Ok(Dtype::F32),
        "float64"           => Ok(Dtype::F64),
        "complex64"         => Ok(Dtype::C64),
        _ => Err(PyTypeError::new_err(
            format!("unsupported dtype: {name}")
        )),
    }
}

pub trait AllowedArrayType<'a, 'py> {
    fn _cast(self) -> PyResult<&'a Bound<'py, PyUntypedArray>>;
}
impl<'a, 'py> AllowedArrayType<'a, 'py> 
for &'a Bound<'py, PyUntypedArray> {
    fn _cast(self) -> PyResult<&'a Bound<'py, PyUntypedArray>> {
        Ok(self)
    }
}
impl<'a, 'py> AllowedArrayType<'a, 'py> 
for &'a Bound<'py, PyAny> {
    fn _cast(self) -> PyResult<&'a Bound<'py, PyUntypedArray>> {
        Ok(self.cast::<PyUntypedArray>()?)
    }
}
pub fn _get_tensor_view<'a, 'py: 'a, T>(array: T) -> anyhow::Result<TensorView<'a>> 
where T: AllowedArrayType<'a, 'py> {
    let new_array: &Bound<'_, PyUntypedArray> = T::_cast(array)?;
    let dtype: Bound<'_, PyArrayDescr> = new_array.dtype();
    let itemsize: usize = dtype.itemsize();
    let byte_len: usize = new_array.len() * itemsize;
    let ptr = unsafe { (*new_array.as_array_ptr()).data as *const u8 };
    let data = unsafe { std::slice::from_raw_parts(ptr, byte_len) };
    Ok(TensorView::new(
        extract_dtype(dtype)?,
        new_array.shape().to_vec(),
        data,
    )?)
}
