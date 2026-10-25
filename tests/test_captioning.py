from imageforge.captioning.base import clean_caption


def test_strips_preamble_and_capitalises():
    assert clean_caption("The image depicts a red barn in snow.") == "A red barn in snow."
    assert clean_caption("the image shows a dog.") == "A dog."
    assert clean_caption("In the image, two bolts rust.") == "Two bolts rust."
    assert clean_caption("The aerial drone photograph shows a heavily damaged area.") == "A heavily damaged area."
    assert clean_caption("The photo features two dogs.") == "Two dogs."
    assert clean_caption("The images of the archive are old.") == "The images of the archive are old."


def test_collapses_whitespace_and_quotes():
    assert clean_caption('  "A  bolt\n on a bench"  ') == "A bolt on a bench"


def test_drops_unfinished_trailing_sentence():
    assert clean_caption("Houses in a grid. Debris on the road. The lighting suggests it") == "Houses in a grid. Debris on the road."
    assert clean_caption("No punctuation at all but one sentence") == "No punctuation at all but one sentence"
    assert clean_caption("Ends cleanly.") == "Ends cleanly."


def test_leaves_plain_text_alone():
    assert clean_caption("A tench held by a fisherman.") == "A tench held by a fisherman."
