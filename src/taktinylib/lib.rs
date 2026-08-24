mod ckpt;
use pyo3::prelude::*;


#[pymodule]
mod taktinylib {
    use pyo3::{exceptions::PyRuntimeError, prelude::*, types::{PyDict, PyList}};
    use pyo3_dlpack::PyTensor;
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
    ) -> PyResult<()>{
        let result = save::_save_safetensors_fn(
            state_dict, 
            path, 
            filename, 
            Some(extension), 
            max_shard_byte_size);

        match result {
            Ok(_) => Ok(()),
            Err(err) => {
                eprintln!("ERROR: {err:#}");
                Err(PyRuntimeError::new_err(format!("{err:#}")))
            },
        }
    }

    #[pyfunction]
    fn foo(py: Python<'_>, tree: &Bound<'_, PyAny>) -> PyResult<()> {
        let tree_util = py.import("jax.tree_util")?;

        // leaves, treedef = jax.tree_util.tree_flatten(tree)
        let flattened = tree_util.call_method1("tree_flatten", (tree,))?;
        let leaves = flattened.get_item(0)?;
        let leaves = leaves.cast::<PyList>()?;

        for leaf in leaves.iter() {
            let tensor = PyTensor::from_pyany(py, &leaf)?;

            println!(
                "shape={:?}, dtype={:?}, device={:?}",
                tensor.shape(),
                tensor.dtype(),
                tensor.device(),
            );
        }

        Ok(())
    }
}