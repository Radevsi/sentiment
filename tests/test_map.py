"""Synthetic-only map tests: no corpus, network or BERT weights."""
import json
import sqlite3
import threading
from urllib.error import HTTPError
from urllib.request import urlopen

import numpy as np
import pytest

from finance_sentiment import map as mapping


@pytest.fixture
def saved_run(tmp_path, monkeypatch):
    root = tmp_path / 'run'
    root.mkdir()
    config = dict(model_id='synthetic', model_revision='a'*40, representation='sum_last_four_cls')
    (root / 'run.json').write_text(json.dumps(dict(config=config)))
    metadata = dict(rows=6, dimensions=4, dtype='float32', **config)
    (root / 'embeddings.json').write_text(json.dumps(metadata))
    np.save(root / 'embeddings.npy', np.arange(24, dtype='float32').reshape(6,4)+1)
    with sqlite3.connect(root / 'counts.sqlite') as conn:
        conn.executescript('''CREATE TABLE vector_rows(row_id INTEGER PRIMARY KEY, text TEXT UNIQUE);
            CREATE TABLE contexts(text TEXT,year INTEGER,match_count INTEGER);
            CREATE TABLE embedding_progress(singleton INTEGER PRIMARY KEY,next_row INTEGER);
            INSERT INTO embedding_progress VALUES(1,6);''')
        # Deliberately nonalphabetical text and reverse insertion order.
        for row in reversed(range(6)):
            text = f'la finance française {5-row} <script>'
            conn.execute('INSERT INTO vector_rows VALUES(?,?)',(row,text))
            for year,count in [(1900,row+1),(1901,10)]:
                conn.execute('INSERT INTO contexts VALUES(?,?,?)',(text,year,count))
    monkeypatch.setattr(mapping, 'versions', lambda: {'synthetic':'1'})
    monkeypatch.setattr(mapping, 'fit_projection', lambda matrix, params: np.asarray(matrix[:, :2]))
    return root, tmp_path / 'map'


def build(saved_run, **kwargs):
    root, output = saved_run
    return mapping.build(root, output, neighbors=2, **kwargs)


def test_all_rows_order_counts_source_untouched_and_cache(saved_run, monkeypatch):
    root, output = saved_run
    before = {p.name: mapping.sha256(p) for p in root.iterdir()}
    info = build(saved_run, expected_rows=6)
    assert info['rows'] == 6 and info['selection'] == 'all'
    assert info['projection']['force_approximation_algorithm'] is True
    assert info['projection']['random_state'] == 42
    assert info['projection']['metric'] == 'cosine'
    np.testing.assert_array_equal(mapping.validate_points(output/'points.f32',6),np.load(root/'embeddings.npy')[:,:2])
    with sqlite3.connect(output/'metadata.sqlite') as conn:
        assert conn.execute('SELECT * FROM points WHERE row_id=0').fetchone() == (0,'la finance française 5 <script>',11,2,1900,1901)
        assert conn.execute('SELECT total_match_count FROM points WHERE row_id=5').fetchone()[0] == 16
    monkeypatch.setattr(mapping,'fit_projection',lambda *a: pytest.fail('Cache must avoid projection'))
    assert build(saved_run)['signature'] == info['signature']
    assert before == {p.name: mapping.sha256(p) for p in root.iterdir()}


def test_completed_projection_resumes_publication(saved_run, monkeypatch):
    build(saved_run)
    (saved_run[1]/'map.json').unlink()
    monkeypatch.setattr(mapping,'fit_projection',lambda *a: pytest.fail('Do not refit completed projection'))
    assert build(saved_run)['rows'] == 6


def test_interrupted_fit_can_restart(saved_run, monkeypatch):
    def crash(*args):
        raise KeyboardInterrupt()
    monkeypatch.setattr(mapping,'fit_projection',crash)
    with pytest.raises(KeyboardInterrupt):
        build(saved_run)
    assert not (saved_run[1]/'map.json').exists()
    monkeypatch.setattr(mapping,'fit_projection',lambda matrix,params: matrix[:,:2])
    assert build(saved_run)['rows'] == 6


@pytest.mark.parametrize('change',['vectors','mapping','counts','settings','packages'])
def test_changed_inputs_rejected_without_overwriting_map(saved_run, monkeypatch, change):
    build(saved_run)
    root, output = saved_run
    before = mapping.sha256(output/'map.json')
    kwargs = {}
    if change == 'vectors':
        matrix = np.load(root/'embeddings.npy'); matrix[0,0] += 1; np.save(root/'embeddings.npy',matrix)
    elif change in ('mapping','counts'):
        with sqlite3.connect(root/'counts.sqlite') as conn:
            if change == 'mapping':
                conn.execute('UPDATE vector_rows SET text=text||" changé" WHERE row_id=0')
                conn.execute('UPDATE contexts SET text=text||" changé" WHERE text LIKE "% 5 %"')
            else:
                conn.execute('UPDATE contexts SET match_count=99 WHERE year=1900')
    elif change == 'settings':
        kwargs['seed']=8
    else:
        monkeypatch.setattr(mapping,'versions',lambda:{'synthetic':'2'})
    with pytest.raises(ValueError,match='changed'):
        build(saved_run,**kwargs)
    assert mapping.sha256(output/'map.json') == before


@pytest.mark.parametrize('change',['gap','missing','incomplete','nonfinite','zero','identity'])
def test_invalid_source_rejected(saved_run,change):
    root, output = saved_run
    if change in ('gap','missing','incomplete'):
        with sqlite3.connect(root/'counts.sqlite') as conn:
            conn.execute({'gap':'UPDATE vector_rows SET row_id=9 WHERE row_id=0',
                          'missing':'DELETE FROM contexts WHERE text LIKE "% 5 %"',
                          'incomplete':'UPDATE embedding_progress SET next_row=5'}[change])
    elif change == 'identity':
        info=json.loads((root/'run.json').read_text()); info['config']['model_revision']='b'*40
        (root/'run.json').write_text(json.dumps(info))
    else:
        matrix=np.load(root/'embeddings.npy');matrix[0]=np.nan if change=='nonfinite' else 0
        np.save(root/'embeddings.npy',matrix)
    with pytest.raises(ValueError):
        build(saved_run)
    assert not (output/'map.json').exists()


def test_bad_projection_and_expected_row_guard(saved_run,monkeypatch):
    with pytest.raises(ValueError,match='Expected'):
        build(saved_run,expected_rows=219584)
    monkeypatch.setattr(mapping,'fit_projection',lambda *a: np.zeros((5,2)))
    with pytest.raises(ValueError,match='invalid coordinates'):
        build(saved_run)
    assert not (saved_run[1]/'map.json').exists()


@pytest.mark.parametrize('name',['points.f32','metadata.sqlite'])
def test_corrupt_artifact_rejected(saved_run,name):
    build(saved_run)
    with (saved_run[1]/name).open('ab') as handle:
        handle.write(b'corrupt')
    with pytest.raises(ValueError,match='checksum'):
        mapping.make_server(saved_run[1],0)


def test_http_points_search_metadata_and_path_isolation(saved_run):
    build(saved_run)
    server=mapping.make_server(saved_run[1],0)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    base=f'http://127.0.0.1:{server.server_port}'
    try:
        assert server.server_address[0]=='127.0.0.1'
        assert len(urlopen(base+'/points.f32').read())==6*8
        result=json.load(urlopen(base+'/api/point?id=5'))
        assert result['row_id']==5 and result['total_match_count']==16
        assert len(json.load(urlopen(base+'/api/find?text=finance')))==6
        assert json.load(urlopen(base+'/api/find?text=%25%25'))==[]  # Literal %, not wildcard.
        for path,code in [('/api/point?id=6',404),('/api/point?id=oops',400),('/api/find?text=a',400),('/../run/counts.sqlite',404),('/metadata.sqlite',404)]:
            with pytest.raises(HTTPError) as error:
                urlopen(base+path)
            assert error.value.code==code
        for path in ['/','/viewer.js','/style.css']:
            response=urlopen(base+path)
            assert response.status==200 and "default-src 'self'" in response.headers['Content-Security-Policy']
    finally:
        server.shutdown();thread.join();server.server_close()


def test_real_umap_smoke_on_synthetic_vectors(tmp_path):
    pytest.importorskip('umap')
    matrix=np.random.default_rng(42).normal(size=(64,8)).astype('float32')
    points=mapping.fit_projection(matrix,dict(n_components=2,metric='cosine',n_neighbors=5,n_epochs=10,
        min_dist=.1,random_state=42,n_jobs=1,low_memory=True,force_approximation_algorithm=True,init='random'))
    assert points.shape==(64,2) and np.isfinite(points).all()
