window.BENCHMARK_DATA = {
  "lastUpdate": 1781283297710,
  "repoUrl": "https://github.com/icecube/pisa",
  "entries": {
    "Benchmark": [
      {
        "commit": {
          "author": {
            "email": "thehrh@users.noreply.github.com",
            "name": "T Ehrhardt",
            "username": "thehrh"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "2c5f2b2b38aea03cd4b62ef6e0447cb9f2101691",
          "message": "Refactor write_benchmark_json to comply with github-action-benchmark format and ensure time series creation (#948)\n\n* instruct documentation workflow to keep_files and set destination_dir to docs\n\n* add entry point for benchmarking script",
          "timestamp": "2026-06-10T19:15:55+02:00",
          "tree_id": "07b52724872421ac05c1fd120b57d188c157a7bb",
          "url": "https://github.com/icecube/pisa/commit/2c5f2b2b38aea03cd4b62ef6e0447cb9f2101691"
        },
        "date": 1781111960449,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "IceCube_3y_neutrinos_daemon (parallel, nthreads=4)",
            "value": 0.8812223064656161,
            "range": "0.24900436401367188",
            "unit": "s",
            "extra": "target=parallel, nthreads=4"
          },
          {
            "name": "IceCube_3y_neutrinos (parallel, nthreads=4)",
            "value": 0.36516611430109763,
            "range": "0.20467233657836914",
            "unit": "s",
            "extra": "target=parallel, nthreads=4"
          },
          {
            "name": "IceCube_3y_muons (parallel, nthreads=4)",
            "value": 0.000498080740169603,
            "range": "0.005239725112915039",
            "unit": "s",
            "extra": "target=parallel, nthreads=4"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "thehrh@users.noreply.github.com",
            "name": "T Ehrhardt",
            "username": "thehrh"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "2c5f2b2b38aea03cd4b62ef6e0447cb9f2101691",
          "message": "Refactor write_benchmark_json to comply with github-action-benchmark format and ensure time series creation (#948)\n\n* instruct documentation workflow to keep_files and set destination_dir to docs\n\n* add entry point for benchmarking script",
          "timestamp": "2026-06-10T19:15:55+02:00",
          "tree_id": "07b52724872421ac05c1fd120b57d188c157a7bb",
          "url": "https://github.com/icecube/pisa/commit/2c5f2b2b38aea03cd4b62ef6e0447cb9f2101691"
        },
        "date": 1781112014924,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "IceCube_3y_neutrinos_daemon (cpu, nthreads=1)",
            "value": 1.228898963149713,
            "range": "0.2408432960510254",
            "unit": "s",
            "extra": "target=cpu, nthreads=1"
          },
          {
            "name": "IceCube_3y_neutrinos (cpu, nthreads=1)",
            "value": 0.9164533371828041,
            "range": "0.1926727294921875",
            "unit": "s",
            "extra": "target=cpu, nthreads=1"
          },
          {
            "name": "IceCube_3y_muons (cpu, nthreads=1)",
            "value": 0.0005603615118532764,
            "range": "0.004967212677001953",
            "unit": "s",
            "extra": "target=cpu, nthreads=1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "thehrh@users.noreply.github.com",
            "name": "T Ehrhardt",
            "username": "thehrh"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "8942e348c62c9590a8f8f2aa5782cee536073d6f",
          "message": "Update data representation management at stage and pipeline level and fix service docs (#934)\n\n* Let Stage class listify supported_reps values if necessary, so that individual services don't need to\n\n* delistify supported_reps values in individual services where possible\n\n* adapt all example pipeline configs and let Pipeline class accept their new configurations\n\n* few small doc fixes\n\n* adapt test_Pipeline() function\n\n* fix bug that allowed running pipeline with invalid (None) stage modes\n\n* new Stage attributes (has_setup, has_compute, has_apply) indicating overriding of abstract base methods\n\n* use new Stage attributes to set supported reps. for given Stage mode by default, unless already set by subclass\n\n* remove explicit definition of supported_reps where not necessary any longer; add docstring note about absence of setup+compute/apply instead\n\n* expand and update service howto and add trace-level logging to Container.mark_changed\n\n* warn when non-trivial supported rep. is detected even though corresponding function isn't implemented (for now, possibly change later)\n\n* auto-generate a service implementation reference table (md or csv) and include md output in stage modes notebook for now\n\n* remove default_translation_mode Container attribute and rename tranlation_modes -> translation_modes; representation management unit test; update container module docs\n\n* remove apply_mode consistency checks at pipeline level and prepare for defining sum_mode_keys in Container\n\n* reintroduce overriding of all binned apply_modes for now, since utils.hist assertion fails otherwise, and adapt pipeline unit test temporarily\n\n* fix Container.__setitem__ and __add_data docstrings and actually auto document all __getitem__ and __setitem__ methods throughout code base",
          "timestamp": "2026-06-12T15:08:44+02:00",
          "tree_id": "3a7c2e27916b4bf3df320ca0cb555bec8c4180f0",
          "url": "https://github.com/icecube/pisa/commit/8942e348c62c9590a8f8f2aa5782cee536073d6f"
        },
        "date": 1781269945137,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "IceCube_3y_neutrinos_daemon (parallel, nthreads=4)",
            "value": 0.8484632044422383,
            "range": "0.19008779525756836",
            "unit": "s",
            "extra": "target=parallel, nthreads=4"
          },
          {
            "name": "IceCube_3y_neutrinos (parallel, nthreads=4)",
            "value": 0.4012960560467778,
            "range": "0.17259478569030762",
            "unit": "s",
            "extra": "target=parallel, nthreads=4"
          },
          {
            "name": "IceCube_3y_muons (parallel, nthreads=4)",
            "value": 0.0005318291333256935,
            "range": "0.0046918392181396484",
            "unit": "s",
            "extra": "target=parallel, nthreads=4"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "thehrh@users.noreply.github.com",
            "name": "T Ehrhardt",
            "username": "thehrh"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "8942e348c62c9590a8f8f2aa5782cee536073d6f",
          "message": "Update data representation management at stage and pipeline level and fix service docs (#934)\n\n* Let Stage class listify supported_reps values if necessary, so that individual services don't need to\n\n* delistify supported_reps values in individual services where possible\n\n* adapt all example pipeline configs and let Pipeline class accept their new configurations\n\n* few small doc fixes\n\n* adapt test_Pipeline() function\n\n* fix bug that allowed running pipeline with invalid (None) stage modes\n\n* new Stage attributes (has_setup, has_compute, has_apply) indicating overriding of abstract base methods\n\n* use new Stage attributes to set supported reps. for given Stage mode by default, unless already set by subclass\n\n* remove explicit definition of supported_reps where not necessary any longer; add docstring note about absence of setup+compute/apply instead\n\n* expand and update service howto and add trace-level logging to Container.mark_changed\n\n* warn when non-trivial supported rep. is detected even though corresponding function isn't implemented (for now, possibly change later)\n\n* auto-generate a service implementation reference table (md or csv) and include md output in stage modes notebook for now\n\n* remove default_translation_mode Container attribute and rename tranlation_modes -> translation_modes; representation management unit test; update container module docs\n\n* remove apply_mode consistency checks at pipeline level and prepare for defining sum_mode_keys in Container\n\n* reintroduce overriding of all binned apply_modes for now, since utils.hist assertion fails otherwise, and adapt pipeline unit test temporarily\n\n* fix Container.__setitem__ and __add_data docstrings and actually auto document all __getitem__ and __setitem__ methods throughout code base",
          "timestamp": "2026-06-12T15:08:44+02:00",
          "tree_id": "3a7c2e27916b4bf3df320ca0cb555bec8c4180f0",
          "url": "https://github.com/icecube/pisa/commit/8942e348c62c9590a8f8f2aa5782cee536073d6f"
        },
        "date": 1781269971544,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "IceCube_3y_neutrinos_daemon (cpu, nthreads=1)",
            "value": 1.1769067705894003,
            "range": "0.18987631797790527",
            "unit": "s",
            "extra": "target=cpu, nthreads=1"
          },
          {
            "name": "IceCube_3y_neutrinos (cpu, nthreads=1)",
            "value": 0.900295656554553,
            "range": "0.15197515487670898",
            "unit": "s",
            "extra": "target=cpu, nthreads=1"
          },
          {
            "name": "IceCube_3y_muons (cpu, nthreads=1)",
            "value": 0.0005517784430056202,
            "range": "0.004430294036865234",
            "unit": "s",
            "extra": "target=cpu, nthreads=1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "32642322+JanWeldert@users.noreply.github.com",
            "name": "Jan Weldert",
            "username": "JanWeldert"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "d6b8a0b73532a4859056b681051199dd03569bd4",
          "message": "Move detectors params handling completely in detectors class (#947)\n\n* streamline detectors class test\n\n* Check params hash before produce outputs\n\n* Init detector class params if values change\n\n* Iterate through self, remove get_hash function and adjust test\n\n* Shorten detectors notebook",
          "timestamp": "2026-06-12T17:07:10+02:00",
          "tree_id": "00dd394e2f6e3b388d991c9df0fe40b5cf85cf5e",
          "url": "https://github.com/icecube/pisa/commit/d6b8a0b73532a4859056b681051199dd03569bd4"
        },
        "date": 1781277053746,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "IceCube_3y_neutrinos_daemon (parallel, nthreads=4)",
            "value": 0.8932430841484849,
            "range": "0.27755260467529297",
            "unit": "s",
            "extra": "target=parallel, nthreads=4"
          },
          {
            "name": "IceCube_3y_neutrinos (parallel, nthreads=4)",
            "value": 0.3662520184808848,
            "range": "0.25457143783569336",
            "unit": "s",
            "extra": "target=parallel, nthreads=4"
          },
          {
            "name": "IceCube_3y_muons (parallel, nthreads=4)",
            "value": 0.0005630862956144372,
            "range": "0.006958484649658203",
            "unit": "s",
            "extra": "target=parallel, nthreads=4"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "32642322+JanWeldert@users.noreply.github.com",
            "name": "Jan Weldert",
            "username": "JanWeldert"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "d6b8a0b73532a4859056b681051199dd03569bd4",
          "message": "Move detectors params handling completely in detectors class (#947)\n\n* streamline detectors class test\n\n* Check params hash before produce outputs\n\n* Init detector class params if values change\n\n* Iterate through self, remove get_hash function and adjust test\n\n* Shorten detectors notebook",
          "timestamp": "2026-06-12T17:07:10+02:00",
          "tree_id": "00dd394e2f6e3b388d991c9df0fe40b5cf85cf5e",
          "url": "https://github.com/icecube/pisa/commit/d6b8a0b73532a4859056b681051199dd03569bd4"
        },
        "date": 1781277091661,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "IceCube_3y_neutrinos_daemon (cpu, nthreads=1)",
            "value": 1.2783815763434585,
            "range": "0.27454638481140137",
            "unit": "s",
            "extra": "target=cpu, nthreads=1"
          },
          {
            "name": "IceCube_3y_neutrinos (cpu, nthreads=1)",
            "value": 0.9110871821033711,
            "range": "0.23987388610839844",
            "unit": "s",
            "extra": "target=cpu, nthreads=1"
          },
          {
            "name": "IceCube_3y_muons (cpu, nthreads=1)",
            "value": 0.000530174800327846,
            "range": "0.005839347839355469",
            "unit": "s",
            "extra": "target=cpu, nthreads=1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "thehrh@users.noreply.github.com",
            "name": "T Ehrhardt",
            "username": "thehrh"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "450871b9424e4b9a24e99154fb5f2492e156fbc4",
          "message": "fetch git tags when checking out repository in actions (#953)",
          "timestamp": "2026-06-12T18:37:38+02:00",
          "tree_id": "04ec0d2c2219ff8be3bb1985e2af8e5a6dd91029",
          "url": "https://github.com/icecube/pisa/commit/450871b9424e4b9a24e99154fb5f2492e156fbc4"
        },
        "date": 1781282479227,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "IceCube_3y_neutrinos_daemon (parallel, nthreads=4)",
            "value": 0.8805331259357686,
            "range": "0.2462468147277832",
            "unit": "s",
            "extra": "target=parallel, nthreads=4"
          },
          {
            "name": "IceCube_3y_neutrinos (parallel, nthreads=4)",
            "value": 0.36479444406470474,
            "range": "0.2171320915222168",
            "unit": "s",
            "extra": "target=parallel, nthreads=4"
          },
          {
            "name": "IceCube_3y_muons (parallel, nthreads=4)",
            "value": 0.0005256253845837652,
            "range": "0.0060520172119140625",
            "unit": "s",
            "extra": "target=parallel, nthreads=4"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "thehrh@users.noreply.github.com",
            "name": "T Ehrhardt",
            "username": "thehrh"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "450871b9424e4b9a24e99154fb5f2492e156fbc4",
          "message": "fetch git tags when checking out repository in actions (#953)",
          "timestamp": "2026-06-12T18:37:38+02:00",
          "tree_id": "04ec0d2c2219ff8be3bb1985e2af8e5a6dd91029",
          "url": "https://github.com/icecube/pisa/commit/450871b9424e4b9a24e99154fb5f2492e156fbc4"
        },
        "date": 1781282522936,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "IceCube_3y_neutrinos_daemon (cpu, nthreads=1)",
            "value": 1.1987008951148208,
            "range": "0.24144387245178223",
            "unit": "s",
            "extra": "target=cpu, nthreads=1"
          },
          {
            "name": "IceCube_3y_neutrinos (cpu, nthreads=1)",
            "value": 0.9234590579052361,
            "range": "0.17775249481201172",
            "unit": "s",
            "extra": "target=cpu, nthreads=1"
          },
          {
            "name": "IceCube_3y_muons (cpu, nthreads=1)",
            "value": 0.0005767540055878308,
            "range": "0.005511045455932617",
            "unit": "s",
            "extra": "target=cpu, nthreads=1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "thehrh@users.noreply.github.com",
            "name": "T Ehrhardt",
            "username": "thehrh"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "2cc03215cfb25ec987ce810ef8f1ba6d96a2985e",
          "message": "Re-synchronise main readme with jupyter notebook generating it and apply some small improvements (#954)\n\n* cosmetic changes\n\n* link to inspirehep instead of arXiv \n\n* improve oscillogram plotting\n\n* refine PISA description",
          "timestamp": "2026-06-12T18:51:30+02:00",
          "tree_id": "dcffd964aa5f2185f722b44eeea74304aac3a973",
          "url": "https://github.com/icecube/pisa/commit/2cc03215cfb25ec987ce810ef8f1ba6d96a2985e"
        },
        "date": 1781283296075,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "IceCube_3y_neutrinos_daemon (parallel, nthreads=4)",
            "value": 0.8471642124409579,
            "range": "0.1874237060546875",
            "unit": "s",
            "extra": "target=parallel, nthreads=4"
          },
          {
            "name": "IceCube_3y_neutrinos (parallel, nthreads=4)",
            "value": 0.3976229064318599,
            "range": "0.16131854057312012",
            "unit": "s",
            "extra": "target=parallel, nthreads=4"
          },
          {
            "name": "IceCube_3y_muons (parallel, nthreads=4)",
            "value": 0.0005512967401621293,
            "range": "0.0052642822265625",
            "unit": "s",
            "extra": "target=parallel, nthreads=4"
          }
        ]
      }
    ]
  }
}