# Copyright 2026 Shinapri
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Hugging Face Hub transport capability for pretrained models."""

from __future__ import annotations

import os
import tempfile
import typing as tp
from collections.abc import Mapping

from huggingface_hub import HfApi

from destiny.utils.typing import PathLike


type Pattern = str | list[str]


class GenericHub:
    """Thin model-Hub transport facade with no checkpoint semantics."""

    _hub_api_type: tp.ClassVar[type[HfApi]] = HfApi
    _hub_library_name: tp.ClassVar[str] = 'destiny'
    _hub_repo_type: tp.ClassVar[str] = 'model'

    @staticmethod
    def _validate_repo_id(repo_id: str) -> str:
        if not isinstance(repo_id, str):
            raise TypeError('repo_id must be a string')
        normalized = repo_id.strip()
        if not normalized:
            raise ValueError('repo_id cannot be empty')
        return normalized

    @classmethod
    def _hub_api(
        cls,
        *,
        token: str | bool | None = None,
        endpoint: str | None = None,
    ) -> HfApi:
        return cls._hub_api_type(
            endpoint=endpoint,
            token=token,
            library_name=cls._hub_library_name,
        )

    @classmethod
    def create_repo(
        cls,
        repo_id: str,
        *,
        private: bool | None = None,
        visibility: str | None = None,
        repo_type: str | None = None,
        exist_ok: bool = False,
        token: str | bool | None = None,
        endpoint: str | None = None,
    ) -> tp.Any:
        """Create a Hub repository and return its repository URL object."""
        repo_id = cls._validate_repo_id(repo_id)
        return cls._hub_api(token=token, endpoint=endpoint).create_repo(
            repo_id=repo_id,
            private=private,
            visibility=visibility,
            repo_type=repo_type or cls._hub_repo_type,
            exist_ok=exist_ok,
            token=token,
        )

    @classmethod
    def repo_info(
        cls,
        repo_id: str,
        *,
        revision: str | None = None,
        repo_type: str | None = None,
        files_metadata: bool = False,
        token: str | bool | None = None,
        endpoint: str | None = None,
    ) -> tp.Any:
        """Return Hub metadata for a repository."""
        repo_id = cls._validate_repo_id(repo_id)
        return cls._hub_api(token=token, endpoint=endpoint).repo_info(
            repo_id=repo_id,
            revision=revision,
            repo_type=repo_type or cls._hub_repo_type,
            files_metadata=files_metadata,
            token=token,
        )

    @classmethod
    def download(
        cls,
        repo_id: str,
        filename: str | None = None,
        *,
        subfolder: str | None = None,
        revision: str | None = None,
        repo_type: str | None = None,
        cache_dir: PathLike | None = None,
        local_dir: PathLike | None = None,
        force_download: bool = False,
        local_files_only: bool = False,
        allow_patterns: Pattern | None = None,
        ignore_patterns: Pattern | None = None,
        max_workers: int = 8,
        dry_run: bool = False,
        token: str | bool | None = None,
        endpoint: str | None = None,
    ) -> tp.Any:
        """Download one file, or a filtered repository snapshot."""
        repo_id = cls._validate_repo_id(repo_id)
        api = cls._hub_api(token=token, endpoint=endpoint)
        common = {
            'repo_id': repo_id,
            'repo_type': repo_type or cls._hub_repo_type,
            'revision': revision,
            'cache_dir': cache_dir,
            'local_dir': local_dir,
            'force_download': force_download,
            'token': token,
            'local_files_only': local_files_only,
            'dry_run': dry_run,
        }
        if filename is not None:
            if not isinstance(filename, str):
                raise TypeError('filename must be a string or None')
            if not filename:
                raise ValueError('filename cannot be empty')
            if allow_patterns is not None or ignore_patterns is not None:
                raise ValueError(
                    'allow_patterns and ignore_patterns apply only to '
                    'snapshot downloads'
                )
            return api.hf_hub_download(
                filename=filename,
                subfolder=subfolder,
                **common,
            )
        if subfolder is not None:
            raise ValueError('subfolder requires a filename')
        if not isinstance(max_workers, int) or isinstance(max_workers, bool):
            raise TypeError('max_workers must be an integer')
        if max_workers < 1:
            raise ValueError('max_workers must be positive')
        return api.snapshot_download(
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
            max_workers=max_workers,
            **common,
        )

    @classmethod
    def upload(
        cls,
        repo_id: str,
        path: PathLike,
        *,
        path_in_repo: str | None = None,
        revision: str | None = None,
        repo_type: str | None = None,
        commit_message: str | None = None,
        commit_description: str | None = None,
        create_pr: bool | None = None,
        parent_commit: str | None = None,
        allow_patterns: Pattern | None = None,
        ignore_patterns: Pattern | None = None,
        delete_patterns: Pattern | None = None,
        run_as_future: bool = False,
        token: str | bool | None = None,
        endpoint: str | None = None,
    ) -> tp.Any:
        """Upload one local file or a directory tree to a Hub repository."""
        repo_id = cls._validate_repo_id(repo_id)
        source = os.path.abspath(os.fspath(path))
        if not os.path.exists(source):
            raise FileNotFoundError(source)
        api = cls._hub_api(token=token, endpoint=endpoint)
        common = {
            'repo_id': repo_id,
            'repo_type': repo_type or cls._hub_repo_type,
            'revision': revision,
            'commit_message': commit_message,
            'commit_description': commit_description,
            'create_pr': create_pr,
            'parent_commit': parent_commit,
            'run_as_future': run_as_future,
            'token': token,
        }
        if os.path.isfile(source):
            if any(
                value is not None
                for value in (
                    allow_patterns,
                    ignore_patterns,
                    delete_patterns,
                )
            ):
                raise ValueError(
                    'upload patterns apply only to directory uploads'
                )
            destination = path_in_repo or os.path.basename(source)
            if not destination:
                raise ValueError('path_in_repo cannot be empty')
            return api.upload_file(
                path_or_fileobj=source,
                path_in_repo=destination,
                **common,
            )
        if path_in_repo == '':
            raise ValueError('path_in_repo cannot be empty')
        return api.upload_folder(
            folder_path=source,
            path_in_repo=path_in_repo,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
            delete_patterns=delete_patterns,
            **common,
        )

    def push_to_hub(
        self,
        repo_id: str,
        *,
        path_in_repo: str | None = None,
        private: bool | None = None,
        revision: str | None = None,
        commit_message: str | None = None,
        commit_description: str | None = None,
        create_pr: bool | None = None,
        parent_commit: str | None = None,
        max_shard_size: int | str = '5GB',
        run_as_future: bool = False,
        token: str | bool | None = None,
        endpoint: str | None = None,
    ) -> tp.Any:
        """Stage ``save_pretrained`` output and upload it to the Hub."""
        repo_id = self._validate_repo_id(repo_id)
        save_pretrained = getattr(self, 'save_pretrained', None)
        if not callable(save_pretrained):
            raise TypeError(
                f'{type(self).__name__} must provide save_pretrained()'
            )
        repository = type(self).create_repo(
            repo_id,
            private=private,
            exist_ok=True,
            token=token,
            endpoint=endpoint,
        )
        resolved_repo_id = getattr(repository, 'repo_id', repo_id)
        api = type(self)._hub_api(token=token, endpoint=endpoint)
        if revision is not None and not revision.startswith('refs/pr/'):
            api.create_branch(
                repo_id=resolved_repo_id,
                branch=revision,
                repo_type=type(self)._hub_repo_type,
                exist_ok=True,
                token=token,
            )

        staging = tempfile.TemporaryDirectory()
        directory = staging.name
        try:
            saved_paths = save_pretrained(
                directory,
                max_shard_size=max_shard_size,
            )
            filenames = {
                os.path.basename(os.fspath(path))
                for path in saved_paths
            }
            is_adapter = any(
                name.startswith('adapter_model')
                for name in filenames
            )
            stem = 'adapter_model' if is_adapter else 'model'
            delete_patterns = [
                f'{stem}.safetensors',
                f'{stem}-*-of-*.safetensors',
                f'{stem}.safetensors.index.json',
            ]
            if not is_adapter:
                delete_patterns.append('quantization_config.json')
            result = type(self).upload(
                resolved_repo_id,
                directory,
                path_in_repo=path_in_repo,
                revision=revision,
                commit_message=commit_message or 'Upload model',
                commit_description=commit_description,
                create_pr=create_pr,
                parent_commit=parent_commit,
                delete_patterns=delete_patterns,
                run_as_future=run_as_future,
                token=token,
                endpoint=endpoint,
            )
        except BaseException:
            staging.cleanup()
            raise

        if run_as_future:
            add_done_callback = getattr(result, 'add_done_callback', None)
            if not callable(add_done_callback):
                staging.cleanup()
                raise TypeError(
                    'asynchronous Hub upload did not return a future'
                )
            add_done_callback(lambda _: staging.cleanup())
            return result
        staging.cleanup()
        return result

    @classmethod
    def from_pretrained(
        cls,
        path_or_repo: PathLike,
        config: tp.Any = None,
        *,
        local: bool = False,
        subfolder: str | None = None,
        config_filename: str = 'config.json',
        weights_filename: str = 'model.safetensors',
        revision: str | None = None,
        cache_dir: PathLike | None = None,
        force_download: bool = False,
        local_files_only: bool = False,
        token: str | bool | None = None,
        endpoint: str | None = None,
        **kwargs: tp.Any,
    ) -> tp.Self:
        """Resolve a local or Hub checkpoint, then delegate materialization."""
        if not isinstance(config_filename, str):
            raise TypeError('config_filename must be a string')
        if not config_filename:
            raise ValueError('config_filename cannot be empty')
        if not isinstance(weights_filename, str):
            raise TypeError('weights_filename must be a string')
        if not weights_filename.endswith('.safetensors'):
            raise ValueError('weights_filename must end with .safetensors')

        source_name = os.fspath(path_or_repo)
        if local:
            root = os.path.abspath(source_name)
            if subfolder:
                root = os.path.join(root, subfolder)
            if not os.path.isdir(root):
                raise NotADirectoryError(root)
        else:
            stem, _ = os.path.splitext(weights_filename)
            prefix = f'{subfolder.rstrip("/")}/' if subfolder else ''
            root = cls.download(
                source_name,
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
                local_files_only=local_files_only,
                allow_patterns=[
                    f'{prefix}{config_filename}',
                    f'{prefix}{weights_filename}',
                    f'{prefix}{weights_filename}.index.json',
                    f'{prefix}{stem}-*.safetensors',
                    f'{prefix}adapter_config.json',
                    f'{prefix}quantization_config.json',
                ],
                token=token,
                endpoint=endpoint,
            )
            if not isinstance(root, str):
                raise TypeError('snapshot download did not return a local path')
            if subfolder:
                root = os.path.join(root, subfolder)

        if isinstance(config, Mapping):
            default_config = getattr(cls, '_default_config', None)
            config_type = type(default_config) if default_config is not None else None
            if config_type is None:
                raise TypeError(
                    f'{cls.__name__} must define _default_config to decode '
                    'configuration mappings'
                )
            config = config_type(**dict(config))
        if config is None:
            default_config = getattr(cls, '_default_config', None)
            config_type = type(default_config) if default_config is not None else None
            if config_type is None:
                raise TypeError(
                    f'{cls.__name__} must define _default_config to load '
                    'pretrained configuration'
                )
            config = config_type.load_config(
                root,
                config_filename,
                local=True,
            )
        materialize = getattr(cls, 'from_pretrained_directory', None)
        if not callable(materialize):
            raise TypeError(
                f'{cls.__name__} must provide from_pretrained_directory()'
            )
        return materialize(
            root,
            config,
            source_name=source_name,
            weights_filename=weights_filename,
            **kwargs,
        )


__all__ = ['GenericHub', 'Pattern']
