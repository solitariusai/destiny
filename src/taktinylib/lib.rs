mod ckpt;
use pyo3::prelude::*;


#[pymodule]
mod taktinylib {
    use pyo3::{exceptions::PyRuntimeError, prelude::*, types::PyDict};
    use std::path::PathBuf;
    use super::ckpt::save;

    /// Formats the sum of two numbers as string.
    #[pyfunction]
    fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
        Ok((a + b).to_string())
    }

    #[pyfunction]
    fn _save_safetensors(
        state_dict: &Bound<'_, PyDict>,
        path: PathBuf,
        filename: &str,
        extension: &str,
        max_shard_byte_size: usize,
    ) -> PyResult<Vec<PathBuf>>{
        let result = save::_save_safetensors_fn(
            state_dict, 
            path, 
            filename, 
            Some(extension), 
            max_shard_byte_size);

        match result {
            Ok(res) => Ok(res),
            Err(err) => {
                eprintln!("ERROR: {err:#}");
                Err(PyRuntimeError::new_err(format!("{err:#}")))
            },
        }
    }

}