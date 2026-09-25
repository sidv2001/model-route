"""Load the public synthetic priors, tasks, and private-to-simulator offsets."""

import json
from dataclasses import dataclass
from importlib import resources
from math import isfinite
from typing import Iterator

from .core import DOMAINS, ModelProfile, Task


def _read_fixture(name: str) -> object:
    path = resources.files("model_route").joinpath("data", name)
    return json.loads(path.read_text(encoding="utf-8"))


def load_profiles() -> tuple[ModelProfile, ...]:
    data = _read_fixture("profiles.json")
    if not isinstance(data, list) or not data:
        raise ValueError("profiles fixture must be a nonempty list")

    profiles = []
    for entry in data:
        if not isinstance(entry, dict) or not isinstance(entry.get("capability_priors"), dict):
            raise ValueError("each profile needs a capability_priors object")
        priors = entry["capability_priors"]
        if any(not isinstance(domain, str) or not isinstance(values, list) for domain, values in priors.items()):
            raise ValueError("capability priors must map domains to lists")
        profiles.append(
            ModelProfile(
                name=entry.get("name"),
                credits_per_call=entry.get("credits_per_call"),
                capability_priors={domain: tuple(values) for domain, values in priors.items()},
            )
        )

    if len({profile.name for profile in profiles}) != len(profiles):
        raise ValueError("profile names must be unique")
    return tuple(profiles)


@dataclass(frozen=True, slots=True)
class Scenario:
    templates: tuple[Task, ...]
    success_offsets: dict[str, dict[str, dict[str, float]]]

    def task_stream(self, rounds: int) -> Iterator[tuple[str, Task]]:
        if type(rounds) is not int or rounds < 1:
            raise ValueError("rounds must be a positive integer")
        index = 0
        for _ in range(rounds):
            for template in self.templates:
                for user_id in self.success_offsets:
                    yield user_id, Task(
                        id=f"task-{index:06d}",
                        domain=template.domain,
                        complexity=template.complexity,
                        description=template.description,
                    )
                    index += 1


def load_scenario(profiles: tuple[ModelProfile, ...]) -> Scenario:
    if not profiles or len({profile.name for profile in profiles}) != len(profiles):
        raise ValueError("scenario requires distinct model profiles")
    data = _read_fixture("scenarios.json")
    if not isinstance(data, dict):
        raise ValueError("scenarios fixture must be an object")
    raw_templates = data.get("task_templates")
    raw_users = data.get("users")
    if not isinstance(raw_templates, list) or not raw_templates:
        raise ValueError("scenarios fixture needs task templates")
    if not isinstance(raw_users, list) or not raw_users:
        raise ValueError("scenarios fixture needs synthetic users")

    templates = []
    for entry in raw_templates:
        if not isinstance(entry, dict) or not isinstance(entry.get("description"), str) or not entry["description"]:
            raise ValueError("each template needs a description")
        templates.append(
            Task(
                id=entry.get("id"),
                domain=entry.get("domain"),
                complexity=entry.get("complexity"),
                description=entry["description"],
            )
        )
    if len({task.id for task in templates}) != len(templates):
        raise ValueError("template ids must be unique")

    offsets: dict[str, dict[str, dict[str, float]]] = {}
    profile_names = {profile.name for profile in profiles}
    for entry in raw_users:
        if not isinstance(entry, dict) or not isinstance(entry.get("id"), str) or not entry["id"]:
            raise ValueError("each synthetic user needs an id")
        user_id = entry["id"]
        if user_id in offsets:
            raise ValueError("synthetic user ids must be unique")
        raw_offsets = entry.get("success_offsets")
        if not isinstance(raw_offsets, dict) or set(raw_offsets) != profile_names:
            raise ValueError("user offsets must cover exactly the model profiles")
        user_offsets = {}
        for profile in profiles:
            per_domain = raw_offsets[profile.name]
            if not isinstance(per_domain, dict) or set(per_domain) != set(DOMAINS):
                raise ValueError("model offsets must cover exactly the task domains")
            for domain in DOMAINS:
                offset = per_domain[domain]
                if (
                    isinstance(offset, bool)
                    or not isinstance(offset, (int, float))
                    or not isfinite(offset)
                    or any(
                        not 0 <= prior + offset <= 1
                        for prior in profile.capability_priors[domain]
                    )
                ):
                    raise ValueError("offsets must produce probabilities in [0, 1]")
            user_offsets[profile.name] = dict(per_domain)
        offsets[user_id] = user_offsets

    return Scenario(tuple(templates), offsets)
