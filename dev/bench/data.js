window.BENCHMARK_DATA = {
  "lastUpdate": 1781111961013,
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
      }
    ]
  }
}