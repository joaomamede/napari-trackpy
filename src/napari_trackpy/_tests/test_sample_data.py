from napari_trackpy._sample_data import make_sample_data


def test_make_sample_data_shape():
    samples = make_sample_data()
    assert isinstance(samples, list)
    assert len(samples) == 1
    data, kwargs = samples[0]
    assert data.shape == (512, 512)
    assert isinstance(kwargs, dict)
