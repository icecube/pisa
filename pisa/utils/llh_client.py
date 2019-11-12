#!/usr/bin/env python

"""
EMCEE-callable cleints for making parallel llh requests to a set of
`simpler_server`s; each client passes free param values to an available server,
server sets these on its DistributionMaker, generates outputs, compares the
resulting distributions against a reference template, and returns the llh value.

`Cleint` code borrowed from Dan Krause
  https://gist.github.com/dankrause/9607475
see `__license__`.
"""


from __future__ import absolute_import, division, print_function


__author__ = "Dan Krause, adapted by J.L. Lanfranchi"

__license__ = """
Copyright 2017 Dan Krause

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License."""

__all__ = ["Client", "get_llh", "setup_sampler", "main"]


from argparse import ArgumentParser
from collections.abc import Mapping
from itertools import cycle
from multiprocessing import Manager
import socket
import time

import emcee
import numpy as np

from pisa.utils.llh_server import send_obj, receive_obj


class Client(object):
    def __init__(self, server_address):
        self.addr = server_address
        if isinstance(self.addr, str):
            address_family = socket.AF_UNIX
        else:
            address_family = socket.AF_INET
        self.sock = socket.socket(address_family, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    def connect(self):
        self.sock.connect(self.addr)

    def close(self):
        self.sock.close()

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def get_llh(self, x):
        send_obj(x, self.sock)
        llh = receive_obj(self.sock)
        return llh


def get_llh(x, server_infos):
    """Get llh given free param values `x` (name chosen for compatibility with
    EMCEE) from a `pisa.utils.llh_server` running somewhere, via TCP-based IPC.

    Parameters
    ----------
    x : sequence
        Free param values to set on the DistributionMaker, at which we wich to
        find llh

    server_infos : dict or iterable thereof
        Each dict must have fields "host", "port", and "lock"

    Returns
    -------
    llh

    """
    if isinstance(server_infos, Mapping):
        server_infos = [server_infos]

    # Cycle through all servers/ports forever
    for server_info in cycle(server_infos):
        if "lock" in server_info:
            if server_info["lock"].acquire(blocking=False):
                try:
                    with Client((server_info["host"], server_info["port"])) as client:
                        llh = client.get_llh(x)
                    print(server_info)
                    return llh
                finally:
                    server_info["lock"].release()
            else:
                # don't hammer ports too hard (not sure about sleep time, though)
                time.sleep(0.1)  # sec
        else:
            with Client((server_info["host"], server_info["port"])) as client:
                llh = client.get_llh(x)
            return llh

    raise ValueError("No hosts?")


def setup_sampler(nwalkers, ndim, host_port_num, **kwargs):
    """Setup/instantiate an `emcee.EnsembleSampler`.

    Parameters
    ----------
    host_port_num : tuple of (host, port, num) or iterable thereof

    nwalkers, ndim, *args, **kwargs
        Passed onto `emcee.EnsembleSampler`; note that fields

            kwargs["threads"]
            kwargs["kwargs"]["server_infos"]

        are overwritten by values derived here (if any of these already exist
        in `kwargs`).

    Returns
    -------
    sampler : emcee.EnsembleSampler

    """
    host_port_num = tuple(host_port_num)
    if isinstance(host_port_num[0], str):
        host_port_num = (host_port_num,)

    # Construct (lock, host, port) dict per port per host, each with a unique lock
    manager = Manager()
    server_infos = []
    for hpn in host_port_num:
        host = str(hpn[0])
        port0 = int(hpn[1])
        num = int(hpn[2])
        for port in range(port0, port0 + num):
            server_infos.append(dict(lock=manager.Lock(), host=host, port=port))
    threads = len(server_infos)

    sub_kwargs = kwargs.get("kwargs", {})
    sub_kwargs["server_infos"] = server_infos
    kwargs["kwargs"] = sub_kwargs

    sampler = emcee.EnsembleSampler(nwalkers, ndim, get_llh, threads=threads, **kwargs)

    return sampler


def main(description=__doc__):
    """Parse command line arguments and execute"""
    parser = ArgumentParser(description=description)
    parser.add_argument(
        "--host-port-num",
        nargs=3,
        action="append",
        metavar="",
        help="""Provide HOST PORT NUM, separated by spaces; repeat
        --host-port-num arg for multiple hosts"""
    )

    kwargs = vars(parser.parse_args())
    ndim = 3
    nwalkers = 100
    sampler = setup_sampler(nwalkers=nwalkers, ndim=ndim, **kwargs)

    rand = np.random.RandomState(0)
    p0 = rand.rand(ndim * nwalkers).reshape((nwalkers, ndim))

    sampler.run_mcmc(p0, nwalkers)


if  __name__ == "__main__":
    main()
