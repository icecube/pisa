window.BENCHMARK_DATA = {
  "lastUpdate": 1785940002855,
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
        "date": 1781283338207,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "IceCube_3y_neutrinos_daemon (cpu, nthreads=1)",
            "value": 1.2162669824094188,
            "range": "0.2122809886932373",
            "unit": "s",
            "extra": "target=cpu, nthreads=1"
          },
          {
            "name": "IceCube_3y_neutrinos (cpu, nthreads=1)",
            "value": 0.8946176645707111,
            "range": "0.16067099571228027",
            "unit": "s",
            "extra": "target=cpu, nthreads=1"
          },
          {
            "name": "IceCube_3y_muons (cpu, nthreads=1)",
            "value": 0.000553199223109654,
            "range": "0.004629373550415039",
            "unit": "s",
            "extra": "target=cpu, nthreads=1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "49699333+dependabot[bot]@users.noreply.github.com",
            "name": "dependabot[bot]",
            "username": "dependabot[bot]"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "2d8cd56ab53c65b329e0473fb86aa20d9aa21107",
          "message": "Bump actions/checkout in /.github/workflows in the all-actions group (#958)\n\nBumps the all-actions group in /.github/workflows with 1 update: [actions/checkout](https://github.com/actions/checkout).\n\n\nUpdates `actions/checkout` from 6 to 7\n- [Release notes](https://github.com/actions/checkout/releases)\n- [Changelog](https://github.com/actions/checkout/blob/main/CHANGELOG.md)\n- [Commits](https://github.com/actions/checkout/compare/v6...v7)\n\n---\nupdated-dependencies:\n- dependency-name: actions/checkout\n  dependency-version: '7'\n  dependency-type: direct:production\n  update-type: version-update:semver-major\n  dependency-group: all-actions\n...\n\nSigned-off-by: dependabot[bot] <support@github.com>\nCo-authored-by: dependabot[bot] <49699333+dependabot[bot]@users.noreply.github.com>",
          "timestamp": "2026-06-23T14:57:58+02:00",
          "tree_id": "7d6d9ce3ffa3617630a4101737150484d2831c69",
          "url": "https://github.com/icecube/pisa/commit/2d8cd56ab53c65b329e0473fb86aa20d9aa21107"
        },
        "date": 1782219652161,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "IceCube_3y_neutrinos_daemon (parallel, nthreads=4)",
            "value": 0.70253950235795,
            "range": "0.21418976783752441",
            "unit": "s",
            "extra": "target=parallel, nthreads=4"
          },
          {
            "name": "IceCube_3y_neutrinos (parallel, nthreads=4)",
            "value": 0.2857263477481141,
            "range": "0.17917108535766602",
            "unit": "s",
            "extra": "target=parallel, nthreads=4"
          },
          {
            "name": "IceCube_3y_muons (parallel, nthreads=4)",
            "value": 0.00039861640151666136,
            "range": "0.0047168731689453125",
            "unit": "s",
            "extra": "target=parallel, nthreads=4"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "49699333+dependabot[bot]@users.noreply.github.com",
            "name": "dependabot[bot]",
            "username": "dependabot[bot]"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "2d8cd56ab53c65b329e0473fb86aa20d9aa21107",
          "message": "Bump actions/checkout in /.github/workflows in the all-actions group (#958)\n\nBumps the all-actions group in /.github/workflows with 1 update: [actions/checkout](https://github.com/actions/checkout).\n\n\nUpdates `actions/checkout` from 6 to 7\n- [Release notes](https://github.com/actions/checkout/releases)\n- [Changelog](https://github.com/actions/checkout/blob/main/CHANGELOG.md)\n- [Commits](https://github.com/actions/checkout/compare/v6...v7)\n\n---\nupdated-dependencies:\n- dependency-name: actions/checkout\n  dependency-version: '7'\n  dependency-type: direct:production\n  update-type: version-update:semver-major\n  dependency-group: all-actions\n...\n\nSigned-off-by: dependabot[bot] <support@github.com>\nCo-authored-by: dependabot[bot] <49699333+dependabot[bot]@users.noreply.github.com>",
          "timestamp": "2026-06-23T14:57:58+02:00",
          "tree_id": "7d6d9ce3ffa3617630a4101737150484d2831c69",
          "url": "https://github.com/icecube/pisa/commit/2d8cd56ab53c65b329e0473fb86aa20d9aa21107"
        },
        "date": 1782219719727,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "IceCube_3y_neutrinos_daemon (cpu, nthreads=1)",
            "value": 1.1903884264887596,
            "range": "0.19089579582214355",
            "unit": "s",
            "extra": "target=cpu, nthreads=1"
          },
          {
            "name": "IceCube_3y_neutrinos (cpu, nthreads=1)",
            "value": 0.8951258805333352,
            "range": "0.16402721405029297",
            "unit": "s",
            "extra": "target=cpu, nthreads=1"
          },
          {
            "name": "IceCube_3y_muons (cpu, nthreads=1)",
            "value": 0.00054694195182956,
            "range": "0.0046918392181396484",
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
          "id": "037293d63017102e4f84b84c9df7357174bc6115",
          "message": "make map plotting compatible with matplotlib 3.11 and add rudimentary unit test (#956)",
          "timestamp": "2026-06-25T19:04:41+02:00",
          "tree_id": "ea8e36df9d5617869c95fec9457d159eb14e7f2f",
          "url": "https://github.com/icecube/pisa/commit/037293d63017102e4f84b84c9df7357174bc6115"
        },
        "date": 1782407295795,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "IceCube_3y_neutrinos_daemon (parallel, nthreads=4)",
            "value": 0.9018552935853297,
            "range": "0.20969247817993164",
            "unit": "s",
            "extra": "target=parallel, nthreads=4"
          },
          {
            "name": "IceCube_3y_neutrinos (parallel, nthreads=4)",
            "value": 0.3981450333887217,
            "range": "0.1595752239227295",
            "unit": "s",
            "extra": "target=parallel, nthreads=4"
          },
          {
            "name": "IceCube_3y_muons (parallel, nthreads=4)",
            "value": 0.0005391957808514031,
            "range": "0.0048656463623046875",
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
          "id": "037293d63017102e4f84b84c9df7357174bc6115",
          "message": "make map plotting compatible with matplotlib 3.11 and add rudimentary unit test (#956)",
          "timestamp": "2026-06-25T19:04:41+02:00",
          "tree_id": "ea8e36df9d5617869c95fec9457d159eb14e7f2f",
          "url": "https://github.com/icecube/pisa/commit/037293d63017102e4f84b84c9df7357174bc6115"
        },
        "date": 1782407332945,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "IceCube_3y_neutrinos_daemon (cpu, nthreads=1)",
            "value": 1.256635417743605,
            "range": "0.2167809009552002",
            "unit": "s",
            "extra": "target=cpu, nthreads=1"
          },
          {
            "name": "IceCube_3y_neutrinos (cpu, nthreads=1)",
            "value": 0.9469128092940973,
            "range": "0.16960501670837402",
            "unit": "s",
            "extra": "target=cpu, nthreads=1"
          },
          {
            "name": "IceCube_3y_muons (cpu, nthreads=1)",
            "value": 0.0005954868939458107,
            "range": "0.005952119827270508",
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
          "id": "8d8ed231fb5b13ce0297f6b40180ec1dbced2d68",
          "message": "Use `ParamSet.priors_penalty` attribute to obtain correct penalty total in all cases (that don't constitute a `Detectors` fit result) (#957)",
          "timestamp": "2026-06-29T11:39:42+02:00",
          "tree_id": "5025f0604ae18c34f2d5fa8f188daba4ff27bc71",
          "url": "https://github.com/icecube/pisa/commit/8d8ed231fb5b13ce0297f6b40180ec1dbced2d68"
        },
        "date": 1782726159759,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "IceCube_3y_neutrinos_daemon (parallel, nthreads=4)",
            "value": 0.7090497406161561,
            "range": "0.21901679039001465",
            "unit": "s",
            "extra": "target=parallel, nthreads=4"
          },
          {
            "name": "IceCube_3y_neutrinos (parallel, nthreads=4)",
            "value": 0.2862099336118114,
            "range": "0.18301725387573242",
            "unit": "s",
            "extra": "target=parallel, nthreads=4"
          },
          {
            "name": "IceCube_3y_muons (parallel, nthreads=4)",
            "value": 0.00040111736375458384,
            "range": "0.004757404327392578",
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
          "id": "8d8ed231fb5b13ce0297f6b40180ec1dbced2d68",
          "message": "Use `ParamSet.priors_penalty` attribute to obtain correct penalty total in all cases (that don't constitute a `Detectors` fit result) (#957)",
          "timestamp": "2026-06-29T11:39:42+02:00",
          "tree_id": "5025f0604ae18c34f2d5fa8f188daba4ff27bc71",
          "url": "https://github.com/icecube/pisa/commit/8d8ed231fb5b13ce0297f6b40180ec1dbced2d68"
        },
        "date": 1782726222086,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "IceCube_3y_neutrinos_daemon (cpu, nthreads=1)",
            "value": 1.2311979848511365,
            "range": "0.18818092346191406",
            "unit": "s",
            "extra": "target=cpu, nthreads=1"
          },
          {
            "name": "IceCube_3y_neutrinos (cpu, nthreads=1)",
            "value": 0.9253183238360346,
            "range": "0.1714932918548584",
            "unit": "s",
            "extra": "target=cpu, nthreads=1"
          },
          {
            "name": "IceCube_3y_muons (cpu, nthreads=1)",
            "value": 0.00054287423892897,
            "range": "0.004860401153564453",
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
          "id": "74fadc6de62f7ba523dffe88accb8801845e7d76",
          "message": "transition to major release tags in docker workflow for consistency (#961)",
          "timestamp": "2026-07-01T15:23:22+02:00",
          "tree_id": "e4464ac8ca79259b9256b211ab7d4b9fcfe2400d",
          "url": "https://github.com/icecube/pisa/commit/74fadc6de62f7ba523dffe88accb8801845e7d76"
        },
        "date": 1782912400099,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "IceCube_3y_neutrinos_daemon (parallel, nthreads=4)",
            "value": 0.8908254078456334,
            "range": "0.18492531776428223",
            "unit": "s",
            "extra": "target=parallel, nthreads=4"
          },
          {
            "name": "IceCube_3y_neutrinos (parallel, nthreads=4)",
            "value": 0.3957405090332031,
            "range": "0.15308022499084473",
            "unit": "s",
            "extra": "target=parallel, nthreads=4"
          },
          {
            "name": "IceCube_3y_muons (parallel, nthreads=4)",
            "value": 0.000545934754974988,
            "range": "0.004652261734008789",
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
          "id": "74fadc6de62f7ba523dffe88accb8801845e7d76",
          "message": "transition to major release tags in docker workflow for consistency (#961)",
          "timestamp": "2026-07-01T15:23:22+02:00",
          "tree_id": "e4464ac8ca79259b9256b211ab7d4b9fcfe2400d",
          "url": "https://github.com/icecube/pisa/commit/74fadc6de62f7ba523dffe88accb8801845e7d76"
        },
        "date": 1782912441499,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "IceCube_3y_neutrinos_daemon (cpu, nthreads=1)",
            "value": 1.2190725657404686,
            "range": "0.2236487865447998",
            "unit": "s",
            "extra": "target=cpu, nthreads=1"
          },
          {
            "name": "IceCube_3y_neutrinos (cpu, nthreads=1)",
            "value": 0.9328256869802669,
            "range": "0.15360784530639648",
            "unit": "s",
            "extra": "target=cpu, nthreads=1"
          },
          {
            "name": "IceCube_3y_muons (cpu, nthreads=1)",
            "value": 0.0005433170162901587,
            "range": "0.004865169525146484",
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
          "id": "902322bcf3e821a5f7a07b019e1fd7ab35329b25",
          "message": "Merge pull request #963 from icecube/mec_sys\n\nMEC sys stage",
          "timestamp": "2026-07-02T13:29:02+02:00",
          "tree_id": "2d25388847835ef764298cce2cd7350355fda36e",
          "url": "https://github.com/icecube/pisa/commit/902322bcf3e821a5f7a07b019e1fd7ab35329b25"
        },
        "date": 1782991944166,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "IceCube_3y_neutrinos_daemon (parallel, nthreads=4)",
            "value": 0.9126433158407405,
            "range": "0.22435355186462402",
            "unit": "s",
            "extra": "target=parallel, nthreads=4"
          },
          {
            "name": "IceCube_3y_neutrinos (parallel, nthreads=4)",
            "value": 0.3779464351887606,
            "range": "0.18012738227844238",
            "unit": "s",
            "extra": "target=parallel, nthreads=4"
          },
          {
            "name": "IceCube_3y_muons (parallel, nthreads=4)",
            "value": 0.0003918725617077886,
            "range": "0.00030231475830078125",
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
          "id": "902322bcf3e821a5f7a07b019e1fd7ab35329b25",
          "message": "Merge pull request #963 from icecube/mec_sys\n\nMEC sys stage",
          "timestamp": "2026-07-02T13:29:02+02:00",
          "tree_id": "2d25388847835ef764298cce2cd7350355fda36e",
          "url": "https://github.com/icecube/pisa/commit/902322bcf3e821a5f7a07b019e1fd7ab35329b25"
        },
        "date": 1782991990098,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "IceCube_3y_neutrinos_daemon (cpu, nthreads=1)",
            "value": 1.2905659480970733,
            "range": "0.23096776008605957",
            "unit": "s",
            "extra": "target=cpu, nthreads=1"
          },
          {
            "name": "IceCube_3y_neutrinos (cpu, nthreads=1)",
            "value": 0.8788976863938935,
            "range": "0.191375732421875",
            "unit": "s",
            "extra": "target=cpu, nthreads=1"
          },
          {
            "name": "IceCube_3y_muons (cpu, nthreads=1)",
            "value": 0.0005228081528021364,
            "range": "0.0065114498138427734",
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
          "id": "d5783d5823ede141425dc428c43b5cf09ac78e3b",
          "message": "prevent vbwkde module test functions from being found by unit testing script (#965)",
          "timestamp": "2026-07-02T15:31:26+02:00",
          "tree_id": "f3497ebe452bb9abe75ade2c5f5bd8b1fde76dc4",
          "url": "https://github.com/icecube/pisa/commit/d5783d5823ede141425dc428c43b5cf09ac78e3b"
        },
        "date": 1782999281679,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "IceCube_3y_neutrinos_daemon (parallel, nthreads=4)",
            "value": 0.8770950375771036,
            "range": "0.19463062286376953",
            "unit": "s",
            "extra": "target=parallel, nthreads=4"
          },
          {
            "name": "IceCube_3y_neutrinos (parallel, nthreads=4)",
            "value": 0.3754215970331309,
            "range": "0.152726411819458",
            "unit": "s",
            "extra": "target=parallel, nthreads=4"
          },
          {
            "name": "IceCube_3y_muons (parallel, nthreads=4)",
            "value": 0.0003995652101477798,
            "range": "0.00021004676818847656",
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
          "id": "d5783d5823ede141425dc428c43b5cf09ac78e3b",
          "message": "prevent vbwkde module test functions from being found by unit testing script (#965)",
          "timestamp": "2026-07-02T15:31:26+02:00",
          "tree_id": "f3497ebe452bb9abe75ade2c5f5bd8b1fde76dc4",
          "url": "https://github.com/icecube/pisa/commit/d5783d5823ede141425dc428c43b5cf09ac78e3b"
        },
        "date": 1782999338457,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "IceCube_3y_neutrinos_daemon (cpu, nthreads=1)",
            "value": 1.2994441888770278,
            "range": "0.23590612411499023",
            "unit": "s",
            "extra": "target=cpu, nthreads=1"
          },
          {
            "name": "IceCube_3y_neutrinos (cpu, nthreads=1)",
            "value": 0.8798423008042939,
            "range": "0.20758438110351562",
            "unit": "s",
            "extra": "target=cpu, nthreads=1"
          },
          {
            "name": "IceCube_3y_muons (cpu, nthreads=1)",
            "value": 0.000518502021322445,
            "range": "0.0066034793853759766",
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
          "id": "e40f7a6782d4b8de1b1785905ebb504edbc8c8d0",
          "message": "ensure pisa version is correctly detected when building documentation, not just for tagged commits (#962)",
          "timestamp": "2026-07-10T14:49:03+02:00",
          "tree_id": "2ee49ebce3e7254702cbcc8f2e04d49f51418cb6",
          "url": "https://github.com/icecube/pisa/commit/e40f7a6782d4b8de1b1785905ebb504edbc8c8d0"
        },
        "date": 1783687944107,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "IceCube_3y_neutrinos_daemon (parallel, nthreads=4)",
            "value": 0.8962589088751345,
            "range": "0.20939159393310547",
            "unit": "s",
            "extra": "target=parallel, nthreads=4"
          },
          {
            "name": "IceCube_3y_neutrinos (parallel, nthreads=4)",
            "value": 0.38419560510285045,
            "range": "0.15701723098754883",
            "unit": "s",
            "extra": "target=parallel, nthreads=4"
          },
          {
            "name": "IceCube_3y_muons (parallel, nthreads=4)",
            "value": 0.00054931640625,
            "range": "0.005742073059082031",
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
          "id": "e40f7a6782d4b8de1b1785905ebb504edbc8c8d0",
          "message": "ensure pisa version is correctly detected when building documentation, not just for tagged commits (#962)",
          "timestamp": "2026-07-10T14:49:03+02:00",
          "tree_id": "2ee49ebce3e7254702cbcc8f2e04d49f51418cb6",
          "url": "https://github.com/icecube/pisa/commit/e40f7a6782d4b8de1b1785905ebb504edbc8c8d0"
        },
        "date": 1783687979665,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "IceCube_3y_neutrinos_daemon (cpu, nthreads=1)",
            "value": 1.2060911704082877,
            "range": "0.23839092254638672",
            "unit": "s",
            "extra": "target=cpu, nthreads=1"
          },
          {
            "name": "IceCube_3y_neutrinos (cpu, nthreads=1)",
            "value": 0.8896897325710375,
            "range": "0.21268272399902344",
            "unit": "s",
            "extra": "target=cpu, nthreads=1"
          },
          {
            "name": "IceCube_3y_muons (cpu, nthreads=1)",
            "value": 0.0006215183102354712,
            "range": "0.00805044174194336",
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
          "id": "a0301b5bb58fd272b6d6ebed30154a3e499f9115",
          "message": "Add new Map properties vmin and vmax and use these to define default colorbar range when 3d map is plotted (#966)\n\n* also adapt Map unit tests",
          "timestamp": "2026-07-16T15:11:47+02:00",
          "tree_id": "cf62efd33a4addf0e6c0816749103a6c3c7cc351",
          "url": "https://github.com/icecube/pisa/commit/a0301b5bb58fd272b6d6ebed30154a3e499f9115"
        },
        "date": 1784207717514,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "IceCube_3y_neutrinos_daemon (parallel, nthreads=4)",
            "value": 0.8964997651625652,
            "range": "0.23303532600402832",
            "unit": "s",
            "extra": "target=parallel, nthreads=4"
          },
          {
            "name": "IceCube_3y_neutrinos (parallel, nthreads=4)",
            "value": 0.3854067422905747,
            "range": "0.17256951332092285",
            "unit": "s",
            "extra": "target=parallel, nthreads=4"
          },
          {
            "name": "IceCube_3y_muons (parallel, nthreads=4)",
            "value": 0.000561514679266482,
            "range": "0.006415843963623047",
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
          "id": "a0301b5bb58fd272b6d6ebed30154a3e499f9115",
          "message": "Add new Map properties vmin and vmax and use these to define default colorbar range when 3d map is plotted (#966)\n\n* also adapt Map unit tests",
          "timestamp": "2026-07-16T15:11:47+02:00",
          "tree_id": "cf62efd33a4addf0e6c0816749103a6c3c7cc351",
          "url": "https://github.com/icecube/pisa/commit/a0301b5bb58fd272b6d6ebed30154a3e499f9115"
        },
        "date": 1784207733738,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "IceCube_3y_neutrinos_daemon (cpu, nthreads=1)",
            "value": 1.102866216581695,
            "range": "0.273775577545166",
            "unit": "s",
            "extra": "target=cpu, nthreads=1"
          },
          {
            "name": "IceCube_3y_neutrinos (cpu, nthreads=1)",
            "value": 0.8000082872351821,
            "range": "0.16198253631591797",
            "unit": "s",
            "extra": "target=cpu, nthreads=1"
          },
          {
            "name": "IceCube_3y_muons (cpu, nthreads=1)",
            "value": 0.0004444122314453125,
            "range": "0.00537109375",
            "unit": "s",
            "extra": "target=cpu, nthreads=1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "49699333+dependabot[bot]@users.noreply.github.com",
            "name": "dependabot[bot]",
            "username": "dependabot[bot]"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "4f80b5f0392312dcb9002139e4eb61976edc7f36",
          "message": "Bump actions/setup-python in /.github/workflows in the all-actions group (#967)\n\nBumps the all-actions group in /.github/workflows with 1 update: [actions/setup-python](https://github.com/actions/setup-python).\n\n\nUpdates `actions/setup-python` from 6 to 7\n- [Release notes](https://github.com/actions/setup-python/releases)\n- [Commits](https://github.com/actions/setup-python/compare/v6...v7)\n\n---\nupdated-dependencies:\n- dependency-name: actions/setup-python\n  dependency-version: '7'\n  dependency-type: direct:production\n  update-type: version-update:semver-major\n  dependency-group: all-actions\n...\n\nSigned-off-by: dependabot[bot] <support@github.com>\nCo-authored-by: dependabot[bot] <49699333+dependabot[bot]@users.noreply.github.com>",
          "timestamp": "2026-07-22T23:23:58+02:00",
          "tree_id": "b3a7199fc0e9f728419131d65c6b6b0835920b57",
          "url": "https://github.com/icecube/pisa/commit/4f80b5f0392312dcb9002139e4eb61976edc7f36"
        },
        "date": 1784755642384,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "IceCube_3y_neutrinos_daemon (parallel, nthreads=4)",
            "value": 0.8961908963261819,
            "range": "0.19539356231689453",
            "unit": "s",
            "extra": "target=parallel, nthreads=4"
          },
          {
            "name": "IceCube_3y_neutrinos (parallel, nthreads=4)",
            "value": 0.3830694081831951,
            "range": "0.1564006805419922",
            "unit": "s",
            "extra": "target=parallel, nthreads=4"
          },
          {
            "name": "IceCube_3y_muons (parallel, nthreads=4)",
            "value": 0.0005707935411102918,
            "range": "0.007082939147949219",
            "unit": "s",
            "extra": "target=parallel, nthreads=4"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "49699333+dependabot[bot]@users.noreply.github.com",
            "name": "dependabot[bot]",
            "username": "dependabot[bot]"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "4f80b5f0392312dcb9002139e4eb61976edc7f36",
          "message": "Bump actions/setup-python in /.github/workflows in the all-actions group (#967)\n\nBumps the all-actions group in /.github/workflows with 1 update: [actions/setup-python](https://github.com/actions/setup-python).\n\n\nUpdates `actions/setup-python` from 6 to 7\n- [Release notes](https://github.com/actions/setup-python/releases)\n- [Commits](https://github.com/actions/setup-python/compare/v6...v7)\n\n---\nupdated-dependencies:\n- dependency-name: actions/setup-python\n  dependency-version: '7'\n  dependency-type: direct:production\n  update-type: version-update:semver-major\n  dependency-group: all-actions\n...\n\nSigned-off-by: dependabot[bot] <support@github.com>\nCo-authored-by: dependabot[bot] <49699333+dependabot[bot]@users.noreply.github.com>",
          "timestamp": "2026-07-22T23:23:58+02:00",
          "tree_id": "b3a7199fc0e9f728419131d65c6b6b0835920b57",
          "url": "https://github.com/icecube/pisa/commit/4f80b5f0392312dcb9002139e4eb61976edc7f36"
        },
        "date": 1784755694810,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "IceCube_3y_neutrinos_daemon (cpu, nthreads=1)",
            "value": 1.2331021124002886,
            "range": "0.223311185836792",
            "unit": "s",
            "extra": "target=cpu, nthreads=1"
          },
          {
            "name": "IceCube_3y_neutrinos (cpu, nthreads=1)",
            "value": 0.9022296350829455,
            "range": "0.17517828941345215",
            "unit": "s",
            "extra": "target=cpu, nthreads=1"
          },
          {
            "name": "IceCube_3y_muons (cpu, nthreads=1)",
            "value": 0.0005478129094960738,
            "range": "0.005466938018798828",
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
          "id": "4f4b9a5efd60ebf3637485cdffb3d91fc2d3f8ad",
          "message": "matplotlib usetex config in main function of scripts.test_flux_weights instead of globally and fall back to mathtext in absence of LaTeX; handle 2015 SNO flux not being part of PISA package (#970)",
          "timestamp": "2026-08-05T16:22:42+02:00",
          "tree_id": "6d7869474b3d87579da9d3d8a168adce21928ea3",
          "url": "https://github.com/icecube/pisa/commit/4f4b9a5efd60ebf3637485cdffb3d91fc2d3f8ad"
        },
        "date": 1785939974962,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "IceCube_3y_neutrinos_daemon (parallel, nthreads=4)",
            "value": 0.9056437502101976,
            "range": "0.26898694038391113",
            "unit": "s",
            "extra": "target=parallel, nthreads=4"
          },
          {
            "name": "IceCube_3y_neutrinos (parallel, nthreads=4)",
            "value": 0.3891011208904033,
            "range": "0.17442727088928223",
            "unit": "s",
            "extra": "target=parallel, nthreads=4"
          },
          {
            "name": "IceCube_3y_muons (parallel, nthreads=4)",
            "value": 0.0005766177663997728,
            "range": "0.006608247756958008",
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
          "id": "4f4b9a5efd60ebf3637485cdffb3d91fc2d3f8ad",
          "message": "matplotlib usetex config in main function of scripts.test_flux_weights instead of globally and fall back to mathtext in absence of LaTeX; handle 2015 SNO flux not being part of PISA package (#970)",
          "timestamp": "2026-08-05T16:22:42+02:00",
          "tree_id": "6d7869474b3d87579da9d3d8a168adce21928ea3",
          "url": "https://github.com/icecube/pisa/commit/4f4b9a5efd60ebf3637485cdffb3d91fc2d3f8ad"
        },
        "date": 1785940001766,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "IceCube_3y_neutrinos_daemon (cpu, nthreads=1)",
            "value": 1.2516897211269455,
            "range": "0.2096412181854248",
            "unit": "s",
            "extra": "target=cpu, nthreads=1"
          },
          {
            "name": "IceCube_3y_neutrinos (cpu, nthreads=1)",
            "value": 0.8890880030028674,
            "range": "0.15626049041748047",
            "unit": "s",
            "extra": "target=cpu, nthreads=1"
          },
          {
            "name": "IceCube_3y_muons (cpu, nthreads=1)",
            "value": 0.0005353567551593391,
            "range": "0.005141019821166992",
            "unit": "s",
            "extra": "target=cpu, nthreads=1"
          }
        ]
      }
    ]
  }
}