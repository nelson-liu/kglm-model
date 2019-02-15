from allennlp.common.util import ensure_list
import numpy as np
import pytest

from kglm.data.dataset_readers import (
    EnhancedWikitextKglmReader,
    EnhancedWikitextEntityNlmReader)


class TestEnhancedWikitextEntityNlmReader:
    @pytest.mark.parametrize('lazy', (True, False))
    def test_read_from_file(self, lazy):
        reader = EnhancedWikitextEntityNlmReader(lazy=lazy)
        fixture_path = 'kglm/tests/fixtures/enhanced-wikitext.jsonl'
        instances = ensure_list(reader.read(fixture_path))

        first_instance_tokens = [x.text for x in instances[0]["tokens"].tokens]
        assert first_instance_tokens[:5] == ['@@START@@', 'State', 'Route', '127', '(']
        assert first_instance_tokens[-5:] == ['Elmer', 'Huntley', 'Bridge', '.', '@@END@@']
        second_instance_entity_types = instances[1]["entity_types"].array
        np.testing.assert_allclose(second_instance_entity_types[:5], [0, 0, 1, 1, 1])
        np.testing.assert_allclose(second_instance_entity_types[-5:], [0, 0, 0, 0, 0])
        np.testing.assert_allclose(instances[1]["entity_ids"].array[:5], [0, 0, 1, 1, 1])
        np.testing.assert_allclose(instances[1]["entity_ids"].array[-5:], [0, 0, 0, 0, 0])
        np.testing.assert_allclose(instances[1]["mention_lengths"].array[:5],
                                   [1, 1, 5, 4, 3])
        np.testing.assert_allclose(instances[1]["mention_lengths"].array[-5:],
                                   [1, 1, 1, 1, 1])

class TestEnhancedWikitextKglmReader:
    @pytest.mark.parametrize('lazy', (True, False))
    def test_read_from_file(self, lazy):
        reader = EnhancedWikitextKglmReader("kglm/tests/fixtures/mini.alias.pkl",
                                            lazy=lazy)
        fixture_path = 'kglm/tests/fixtures/enhanced-wikitext.jsonl'
        instances = ensure_list(reader.read(fixture_path))

        # TODO (nfliu): Ask robert if we want to standardize on source or tokens?
        first_instance_source = [x.text for x in instances[0]["source"].tokens]
        assert first_instance_source[:5] == ['@@START@@', 'State', 'Route', '127', '(']
        assert first_instance_source[-5:] == ['the', 'Elmer', 'Huntley', 'Bridge', '.']
        first_instance_entity_ids = [x.text for x in instances[0]["entity_ids"].tokens]
        assert first_instance_entity_ids == [
            '@@PADDING@@', '@@PADDING@@', '@@PADDING@@', '@@PADDING@@', '@@PADDING@@',
            '@@PADDING@@', '@@PADDING@@', '@@PADDING@@', '@@PADDING@@', '@@PADDING@@',
            '@@PADDING@@', '@@PADDING@@', '@@PADDING@@', '@@PADDING@@', '@@PADDING@@',
            '@@PADDING@@', 'Q831285', 'Q831285', '@@PADDING@@', '@@PADDING@@', 'Q3046581',
            'Q3046581', '@@PADDING@@', '@@PADDING@@', 'Q35657', 'Q35657', '@@PADDING@@',
            'Q1223', '@@PADDING@@', '@@PADDING@@', '@@PADDING@@', '@@PADDING@@',
            '@@PADDING@@', '@@PADDING@@', '@@PADDING@@', 'Q800459', 'Q800459', 'Q800459',
            '@@PADDING@@', '@@PADDING@@', '@@PADDING@@', '@@PADDING@@', 'Q695782',
            'Q695782', '@@PADDING@@', 'Q452623', 'Q452623', 'Q452623', '@@PADDING@@',
            '@@PADDING@@', '@@PADDING@@', '@@PADDING@@', '@@PADDING@@', '@@PADDING@@',
            '@@PADDING@@', '@@PADDING@@', '@@PADDING@@', 'Q272074', 'Q272074',
            '@@PADDING@@', '@@PADDING@@', '@@PADDING@@', '@@PADDING@@', '@@PADDING@@',
            '@@PADDING@@', '@@PADDING@@']
        np.testing.assert_allclose(instances[0]["alias_copy_inds"].array,
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                                    0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0,
                                    0, 1, 2, 3, 0, 0, 0, 0, 1, 2, 0, 1, 2, 3, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0])
        first_instance_shortlist = [x.text for x in instances[0]["shortlist"].tokens]
        assert first_instance_shortlist == [
            '@@PADDING@@', 'Q831285', 'Q3046581', 'Q35657', 'Q1223', 'Q800459',
            'Q695782', 'Q452623', 'Q272074']
        np.testing.assert_allclose(instances[0]["shortlist_inds"].array,
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                                    1, 0, 0, 2, 2, 0, 0, 3, 3, 0, 4, 0, 0, 0, 0, 0, 0,
                                    0, 5, 5, 5, 0, 0, 0, 0, 6, 6, 0, 7, 7, 7, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 8, 8, 0, 0, 0, 0, 0, 0, 0])
        first_instance_parent_ids = [x.text for x in instances[0]["parent_ids"].tokens]
        assert first_instance_parent_ids == [
            'Q831285', 'Q3046581', 'Q35657', 'Q831285', 'Q3046581', 'Q800459', 'Q1223',
            'Q1223', 'Q695782', 'Q272074']
        first_instance_relations = [x.text for x in instances[0]["relations"].tokens]
        assert first_instance_relations == [
            '@@NEW@@', '@@NEW@@', '@@NEW@@', 'P131', 'P131', '@@NEW@@', 'P150',
            'R:P131', 'R:P131', '@@NEW@@']
        np.testing.assert_allclose(instances[0]["has_parent_mask"].array,
                                   [0, 0, 0, 1, 1, 0, 1, 1, 1, 0])
