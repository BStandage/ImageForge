"""Turn captions into generation prompts."""

from dataclasses import dataclass

from .dataset import pretty_class_name


@dataclass
class Caption:
    # Path of the real image this caption describes. None for a synthesised description.
    path: str | None
    class_name: str
    caption: str
    source: str = "real"


class PromptBank:
    """Per-class pool of captions plus the templating that wraps them.

    `references` lists the real image paths per class that can serve as style references.
    By default these are the paths of the real captions in the pool.
    """

    def __init__(
        self,
        cfg,
        captions,
        references=None,
    ):
        self.cfg = cfg
        self.by_class = {}
        for c in captions:
            self.by_class.setdefault(c.class_name, []).append(c)
        if references is None:
            references = {}
            for c in captions:
                if c.path:
                    references.setdefault(c.class_name, []).append(c.path)
        self.references = references

    def classes(self):
        return sorted(self.by_class)

    def render(self, class_name, caption, rng):
        template = self.cfg.class_templates.get(class_name, self.cfg.template)
        style = rng.choice(self.cfg.styles) if self.cfg.styles else ""
        text = template.format(
            class_name=pretty_class_name(class_name),
            caption=caption.caption,
            style=style,
        )
        return " ".join(text.split())

    def sample_captions(self, class_name, n, rng):
        """n captions for a class, cycling through the pool evenly before repeating."""
        pool = self.by_class.get(class_name)
        if not pool:
            raise ValueError(f"no captions for class {class_name!r}; run describe first")
        out = []
        while len(out) < n:
            order = list(pool)
            rng.shuffle(order)
            out.extend(order[: n - len(out)])
        return out

    def pick_reference(self, class_name, caption, pairing, rng):
        """A real image path to condition on for this caption."""
        if caption.path and pairing == "same":
            return caption.path
        pool = self.references.get(class_name, [])
        candidates = [p for p in pool if p != caption.path] or pool
        if not candidates:
            raise ValueError(f"no real reference images for class {class_name!r}")
        return rng.choice(candidates)
