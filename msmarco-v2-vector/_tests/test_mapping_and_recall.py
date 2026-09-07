# Licensed to Elasticsearch B.V. under one or more contributor
# license agreements. See the NOTICE file distributed with
# this work for additional information regarding copyright
# ownership. Elasticsearch B.V. licenses this file to you under
# the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import importlib.util
import json
import pathlib
import types
import unittest.mock

import jinja2
import pytest

TRACK_DIR = pathlib.Path(__file__).parents[1]

_spec = importlib.util.spec_from_file_location("msmarco_v2_vector_track_mapping", TRACK_DIR / "track.py")
track_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(track_module)


def fake_track(index_name="msmarco-v2"):
    return types.SimpleNamespace(indices=[types.SimpleNamespace(name=index_name)])


def render_mapping(template_name, params=None):
    env = jinja2.Environment(loader=jinja2.FileSystemLoader(str(TRACK_DIR)))
    base = {
        "build_flavor": "serverless",  # suppresses non-serverless settings for simpler JSON
        "preload_pagecache": False,
        "aggressive_merge_policy": False,
        "enable_experimental_features": False,
    }
    if params:
        base.update(params)
    rendered = env.get_template(template_name).render(**base)
    return json.loads(rendered)


def render_operations(params=None):
    ops_dir = TRACK_DIR / "operations"
    env = jinja2.Environment(loader=jinja2.FileSystemLoader(str(ops_dir)))
    base = {"index_settings": {}}
    if params:
        base.update(params)
    rendered = env.get_template("default.json").render(**base)
    return json.loads(f"[{rendered}]")


# --- dense_vector index_options: auto_calibrate ---


@pytest.mark.parametrize(
    "template",
    ["index-vectors-only-mapping.json", "index-vectors-with-text-mapping.json"],
)
class TestMappingAutoCalibrate:
    def test_auto_calibrate_true_included(self, template):
        mapping = render_mapping(template, {"auto_calibrate": True})
        index_options = mapping["mappings"]["properties"]["emb"]["index_options"]
        assert index_options.get("auto_calibrate") is True

    def test_auto_calibrate_false_included(self, template):
        mapping = render_mapping(template, {"auto_calibrate": False})
        index_options = mapping["mappings"]["properties"]["emb"]["index_options"]
        assert index_options.get("auto_calibrate") is False

    def test_auto_calibrate_absent_by_default(self, template):
        mapping = render_mapping(template)
        index_options = mapping["mappings"]["properties"]["emb"]["index_options"]
        assert "auto_calibrate" not in index_options

    def test_auto_calibrate_coexists_with_hnsw_params(self, template):
        mapping = render_mapping(template, {"hnsw_m": 16, "hnsw_ef_construction": 100, "auto_calibrate": True})
        index_options = mapping["mappings"]["properties"]["emb"]["index_options"]
        assert index_options["m"] == 16
        assert index_options["ef_construction"] == 100
        assert index_options["auto_calibrate"] is True


# --- KnnRecallParamSource: recall-doc-set propagation ---


class TestKnnRecallParamSource:
    def test_default_recall_doc_set_is_sentinel(self):
        ps = track_module.KnnRecallParamSource(fake_track(), {})
        assert ps.params()["recall_doc_set"] == -1

    def test_10m_recall_doc_set_propagated(self):
        ps = track_module.KnnRecallParamSource(fake_track(), {"recall-doc-set": "10m"})
        assert ps.params()["recall_doc_set"] == "10m"

    def test_full_recall_doc_set_propagated(self):
        ps = track_module.KnnRecallParamSource(fake_track(), {"recall-doc-set": "full"})
        assert ps.params()["recall_doc_set"] == "full"


# --- KnnRecallRunner: query file selection ---


def _make_bz2_mock():
    """Context-manager mock for bz2.open that yields an empty iterator."""
    cm = unittest.mock.MagicMock()
    cm.__enter__ = unittest.mock.Mock(return_value=iter([]))
    cm.__exit__ = unittest.mock.Mock(return_value=False)
    return cm


_RUNNER_PARAMS = {
    "index": "test",
    "cache": False,
    "size": 10,
    "num_candidates": 100,
    "visit_percentage": -1,
    "oversample_rescore": -1,
}


@pytest.mark.asyncio
async def test_runner_10m_opens_10m_queries_file():
    opened = []
    bz2_mock = _make_bz2_mock()

    def fake_open(path, mode):
        opened.append(path)
        return bz2_mock

    with (
        unittest.mock.patch("bz2.open", fake_open),
        unittest.mock.patch.object(track_module, "calc_ndcg", return_value={"ndcg_cut@10": 0.0}),
    ):
        runner = track_module.KnnRecallRunner()
        await runner(None, {**_RUNNER_PARAMS, "recall_doc_set": "10m"})

    assert any(track_module.QUERIES_RECALL_10M_FILENAME in p for p in opened)
    assert not any(track_module.QUERIES_RECALL_FILENAME == p.split("/")[-1] for p in opened)


@pytest.mark.asyncio
async def test_runner_full_opens_standard_queries_file():
    opened = []
    bz2_mock = _make_bz2_mock()

    def fake_open(path, mode):
        opened.append(path)
        return bz2_mock

    with (
        unittest.mock.patch("bz2.open", fake_open),
        unittest.mock.patch.object(track_module, "calc_ndcg", return_value={"ndcg_cut@10": 0.0}),
    ):
        runner = track_module.KnnRecallRunner()
        await runner(None, {**_RUNNER_PARAMS, "recall_doc_set": "full"})

    assert any(track_module.QUERIES_RECALL_FILENAME in p for p in opened)
    assert not any(track_module.QUERIES_RECALL_10M_FILENAME in p for p in opened)


@pytest.mark.asyncio
async def test_runner_default_sentinel_opens_standard_queries_file():
    opened = []
    bz2_mock = _make_bz2_mock()

    def fake_open(path, mode):
        opened.append(path)
        return bz2_mock

    with (
        unittest.mock.patch("bz2.open", fake_open),
        unittest.mock.patch.object(track_module, "calc_ndcg", return_value={"ndcg_cut@10": 0.0}),
    ):
        runner = track_module.KnnRecallRunner()
        await runner(None, {**_RUNNER_PARAMS, "recall_doc_set": -1})

    assert any(track_module.QUERIES_RECALL_FILENAME in p for p in opened)
    assert not any(track_module.QUERIES_RECALL_10M_FILENAME in p for p in opened)


# --- operations/default.json: recall-doc-set based on initial_indexing_ingest_doc_count ---


class TestOperationsRecallDocSet:
    def _recall_doc_sets(self, params=None):
        ops = render_operations(params)
        return {o["recall-doc-set"] for o in ops if o.get("operation-type") == "knn-recall"}

    def test_default_is_full(self):
        assert self._recall_doc_sets() == {"full"}

    def test_10m_doc_count_selects_10m_set(self):
        assert self._recall_doc_sets({"initial_indexing_ingest_doc_count": 10_000_000}) == {"10m"}

    def test_other_doc_count_selects_full_set(self):
        assert self._recall_doc_sets({"initial_indexing_ingest_doc_count": 5_000_000}) == {"full"}

    def test_all_knn_recall_ops_get_same_doc_set(self):
        ops = render_operations({"initial_indexing_ingest_doc_count": 10_000_000})
        recall_ops = [o for o in ops if o.get("operation-type") == "knn-recall"]
        assert len(recall_ops) > 0
        assert all(o["recall-doc-set"] == "10m" for o in recall_ops)
