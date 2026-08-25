use pyo3::{prelude::*, types::PyDict};
use std::{
    collections::HashMap,
    fs::File,
    io::Write,
    path::PathBuf
};
use safetensors::serialize_to_file;
use numpy::{
    PyArrayDescr, 
    PyArrayDescrMethods, 
    PyUntypedArray, 
    PyUntypedArrayMethods,
};
use super::utils::_get_tensor_view;
use std::fmt::Display;


fn _get_prefix_id<T: Display>(id: T) -> String {
    let idstr = id.to_string();
    let remain = 5 - idstr.len();
    let prefix_zeros = "0".repeat(remain);
    format!("{prefix_zeros}{idstr}")
}

fn _filename_condition(
    filename: &str, 
    shard_idx: u32,
    extension: Option<&str>,
    num_shards: usize,
) -> String {
    let extension = extension.unwrap_or("safetensors");
    let id = _get_prefix_id(shard_idx + 1);
    let n = _get_prefix_id(num_shards);
    format!("{filename}-{id}-of-{n}.{extension}")
}

fn _insert_layer_idx(key: String, idx: usize) -> String{
    key.replace("stacked", &idx.to_string())
}

pub fn _save_safetensors_fn(
    state_dict: &Bound<'_, PyDict>, 
    path: PathBuf,
    filename: &str,
    extension: Option<&str>,
    max_shard_byte_size: usize,
) -> anyhow::Result<Vec<PathBuf>>{
    let mut arrays: HashMap<String, Bound<'_, PyUntypedArray>> = HashMap::new();
    let mut weight_map: HashMap<String, String> = HashMap::new();
    let mut shard_idx: u32 = 0;
    let mut accm_byte_size: usize = 0;
    let mut filename_with_extension: String;
    let extension = extension.unwrap_or("safetensors");
    let tmpdir = &path;
    let mut dtype: Bound<'_, PyArrayDescr>;
    let mut paths: Vec<PathBuf> = vec![];
    let mut filepath: PathBuf;

    let mut num_shards: usize = 1;
    for (_key, _arrays) in state_dict.iter() {
        let _arrays: Bound<'_, pyo3::PyAny> = 
            match _arrays.call_method0("__array__") {
                Ok(array) => array,
                Err(err) => {
                    eprintln!("cast failed: {err}");
                    return Err(anyhow::anyhow!(err.to_string()));
                }
            };
        let _arrays: Bound<'_, PyUntypedArray> = 
            match _arrays.cast_into::<PyUntypedArray>() {
                Ok(array) => array,
                Err(err) => {
                    eprintln!("cast failed: {err}");
                    return Err(anyhow::anyhow!(err.to_string()));
                }
            };
        dtype = _arrays.dtype();
        accm_byte_size += _arrays.len() * dtype.itemsize();
    }

    if accm_byte_size > max_shard_byte_size {
        num_shards = accm_byte_size.div_ceil(max_shard_byte_size);
    }

    accm_byte_size = 0;
    for (_key, _arrays) in state_dict.iter(){
        
        let _key: String = match _key.extract() {
            Ok(k) => k,
            Err(err) => {
                eprintln!("failed: {err}");
                return Err(anyhow::anyhow!(err.to_string()));
            }
        };
        // this safe for both jax array and numpy array
        let _arrays: Bound<'_, pyo3::PyAny> = 
            match _arrays.call_method0("__array__") {
                Ok(array) => array,
                Err(err) => {
                    eprintln!("cast failed: {err}");
                    return Err(anyhow::anyhow!(err.to_string()));
                }
            };
        let _arrays: Bound<'_, PyUntypedArray> = 
            match _arrays.cast_into::<PyUntypedArray>() {
                Ok(array) => array,
                Err(err) => {
                    eprintln!("cast failed: {err}");
                    return Err(anyhow::anyhow!(err.to_string()));
                }
            };

        let dtype: Bound<'_, PyArrayDescr> = _arrays.dtype();
        if _key.contains("stacked") {
            let num_stacks: usize = _arrays.shape()[0];

            for i in 0..num_stacks {
                filename_with_extension = _filename_condition(
                    filename, shard_idx, Some(extension), num_shards);
                
                let _array: Bound<'_, PyUntypedArray> = 
                    _arrays.get_item(i)?
                    .cast_into::<PyUntypedArray>()
                    .map_err(PyErr::from)?;

                weight_map.insert(
                    _insert_layer_idx(_key.clone(), i), 
                    filename_with_extension.clone());

                if accm_byte_size > max_shard_byte_size {
                    // _ = serialize_to_file(
                    //     arrays.iter().map(|(k, v)| {
                    //         Ok((k, _get_tensor_view(v)?))
                    //     }).collect::<anyhow::Result<Vec<_>>>()?, None, 
                    //     &tmpdir.join(&filename_with_extension)
                    // );
                    filepath = _serialize_to_file(
                        tmpdir.join(&filename_with_extension), 
                        &arrays)?;
                    paths.push(filepath);
                    arrays.clear();
                    accm_byte_size = 0;
                    shard_idx += 1;
                }
                accm_byte_size += _array.len() * dtype.itemsize();
                arrays.insert(_key.clone(), _array);
            }
            continue;
        }

        filename_with_extension = _filename_condition(
            filename, shard_idx, Some(extension), num_shards);

        weight_map.insert(_key.clone(), filename_with_extension.clone());
        if accm_byte_size > max_shard_byte_size {
            // _ = serialize_to_file(
            //     arrays.iter().map(|(k, v)| {
            //         Ok((k, _get_tensor_view(v)?))
            //     }).collect::<anyhow::Result<Vec<_>>>()?, None, 
            //     &tmpdir.join(&filename_with_extension)
            // );
            filepath = _serialize_to_file(
                tmpdir.join(&filename_with_extension), 
                &arrays)?;
            paths.push(filepath);
            arrays.clear();
            accm_byte_size = 0;
            shard_idx += 1;
        }

        accm_byte_size += _arrays.len() * dtype.itemsize();
        arrays.insert(_key.clone(), _arrays);

    }

    if !arrays.is_empty() {
        let mut final_filename = format!("{filename}.{extension}");
        if num_shards > 1 {
            final_filename = _filename_condition(
                filename, shard_idx, Some(extension), num_shards);
        }
        // _ = serialize_to_file(
        //     arrays.iter().map(|(k, v)| {
        //         Ok((k, _get_tensor_view(v)?))
        //     }).collect::<anyhow::Result<Vec<_>>>()?, None, 
        //     &tmpdir.join(&final_filename)
        // );
        filepath = _serialize_to_file(
            tmpdir.join(&final_filename), 
            &arrays)?;
        paths.push(filepath);
    }

    if num_shards > 1 {
        // let shard_files: HashSet<String> = weight_map
        //     .values().cloned().collect();
        // let idstr = _get_prefix_id(shard_idx + 1);
        // for v in weight_map.values_mut() {
        //     *v = v.replace("PLACEHOLDER", &idstr);
        // }

        let weight_map_json: String = 
            serde_json::to_string_pretty(&weight_map)?;
        let index_filename_with_extension = 
            format!("{filename}.{extension}.index.json");
        let index_file_path = 
            tmpdir.join(&index_filename_with_extension);
        let mut index_file: File = 
            File::create(&index_file_path)?;

        index_file.write_all(weight_map_json.as_bytes())?;
        paths.push(index_file_path);
        
        // for file in shard_files {
        //     let source = tmpdir.join(&file);
        //     let destination = path.join(
        //         &file.replace("PLACEHOLDER", &idstr));
        //     _ = _move_file(source, destination)?;
        // }
        // _ = _move_file(
        //     index_file_path, 
        //     path.join(&index_filename_with_extension))?;
    }

    Ok(paths)
}

fn _serialize_to_file(
    path: PathBuf, 
    state_dict: &HashMap<String, Bound<'_, PyUntypedArray>>
) -> anyhow::Result<PathBuf> {
    _ = serialize_to_file(
        state_dict.iter().map(|(k, v)| {
            Ok((k, _get_tensor_view(v)?))
        }).collect::<anyhow::Result<Vec<_>>>()?, None, 
        &path
    );
    Ok(path)
}

// fn _move_file(
//     source: PathBuf, 
//     destination: PathBuf
// ) -> anyhow::Result<()>{
//     match std::fs::rename(&source, &destination) {
//         Ok(_) => anyhow::Ok(()),
//         Err(_) => {
//             // std::fs::copy(&source, &destination)?;
//             // std::fs::remove_file(&source)?;
//             anyhow::Ok(())
//         }
//     }
// }