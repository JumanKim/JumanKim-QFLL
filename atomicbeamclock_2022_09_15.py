# 2022.09.05 by Juman

import numpy as np
from numba import njit, prange, jit
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import random
import os
from scipy import fftpack
from scipy.optimize import curve_fit  # https://smlee729.github.io/python/simulation/2015/03/25/2-curve_fitting.html
from multiprocessing import Pool, Queue
import multiprocessing
import warnings
from tqdm.gui import trange
warnings.filterwarnings("ignore")
import matplotlib
from tqdm import tqdm
# from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing.pool import ThreadPool
from concurrent.futures import ThreadPoolExecutor
from PyQt5.QtWidgets import *
from PyQt5.QtCore import pyqtSlot, QObject, pyqtSignal, QThread, QThreadPool, QRunnable, QTimer
import logging
# multiprocessing.set_start_method('fork')
logging.basicConfig(format="%(message)s", level=logging.INFO)
_STOPPED = False
pbar = tqdm(total=100000)
_PROGRESS_STATUS = multiprocessing.Value('i', 0)

class WorkerSignals(QObject):
    finished_signal = pyqtSignal()
    progress_signal = pyqtSignal(float)
    prep_signal = pyqtSignal(int, int)
    kappatau = pyqtSignal(float)
    kappagsqrt = pyqtSignal(float)
    kappadelD = pyqtSignal(float)
    NtauGammac = pyqtSignal(float)
    unittime = pyqtSignal(float)
    longtimeerror = pyqtSignal(str)


class Sngltraj_pool():
    signal = pyqtSignal(int, int)
    def __init__(self, ntraj, atomtype, cs, clusnum, kappa, g, tau, ctlth, stpsize, cstpsiz, dt, delDtau, delD, delT, delca, delpa, rhoee, \
        pumplinewidth, sqrtFWHM, accumulated_phase, manipulated_var, dependent_var, randomphase, fftcalc, fftplot, fftfit, theta):
        super().__init__()
        self.cs = cs
        self.clusnum = clusnum
        self.clusnum_list = np.linspace(5, self.clusnum, 20).astype(int)
        self.kappa = kappa
        self.g = g
        self.tau = tau
        self.ctlth = ctlth
        self.stpsize = stpsize
        self.cstpsiz = cstpsiz
        self.dt = self.tau / 200
        self.delDtau = delDtau
        self.delD = self.delDtau / self.tau
        self.delT = delT
        self.delca = delca
        self.delpa = delpa
        self.dca_list = np.linspace(-2 * self.delca, 0.0, 50)
        self.dpa_list = np.linspace(-self.delpa, self.delpa, 50)
        self.rhoee = rhoee
        self.theta = 2.0 * np.arcsin(np.sqrt(self.rhoee))
        self.pumplinewidth = pumplinewidth
        self.sqrtFWHM = np.sqrt(self.pumplinewidth)
        self.accumulated_phase = 0.0
        self.manipulated_var = manipulated_var
        self.dependent_var = dependent_var
        self.randomphase = randomphase
        self.fftcalc = fftcalc
        self.fftplot = fftplot
        self.fftfit = fftfit
        global _STOPPED

    def run(self, n):
        cs, clusnum, kappa, g, tau, ctlth, stpsize, cstpsiz, dt, delDtau, delD, delT, delca, delpa, rhoee, \
        pumplinewidth, sqrtFWHM, accumulated_phase, manipulated_var, dependent_var = self.cs, self.clusnum, self.kappa, self.g, \
                                                                                     self.tau, self.ctlth, self.stpsize, \
                                                                                     self.cstpsiz, self.dt, self.delDtau, \
                                                                                     self.delD, self.delT, self.delca, self.delpa, \
                                                                                     self.rhoee, self.pumplinewidth, self.sqrtFWHM, \
                                                                                     self.accumulated_phase, self.manipulated_var, \
                                                                                     self.dependent_var
        randomphase = self.randomphase
        fftcalc = self.fftcalc
        fftplot = self.fftplot
        fftfit = self.fftfit
        theta = self.theta

        if dependent_var == 'evolve':
            t_final = 5 * tau  # microsec, simulation time
            stpsize = 1
        elif dependent_var == 'outputpower':
            t_final = 500 * tau  # microsec, simulation time
            stpsize = 1
        else:
            t_final = 10000  # microsec, simulation time

        t_length = int(t_final / dt)
        t_list = np.linspace(0, t_final, t_length)
        # t_length = 10000
        # t_final = 10
        clusnum = self.clusnum_list[n]
        # clusnum_list = np.array([clusnum])
        dca_list = np.linspace(-2 * delca, 0.0, 50)
        dpa_list = np.linspace(-delpa, delpa, 50)
        Gammac = g ** 2.0 * kappa / 4.0 / (kappa ** 2.0 / 4.0 + delca ** 2.0)
        Gamma0 = g ** 2.0 / kappa
        GammaD = g ** 2.0 * delca / 2.0 / (kappa ** 2.0 / 4.0 + delca ** 2.0)
        sx = np.full(clusnum, 0.0)
        sy = np.full(clusnum, 0.0)
        sz = np.full(clusnum, 0.0)
        eta = np.full(clusnum, 0.0)
        z = np.full(clusnum, 0.0)  # initial injection location
        vz = np.full(clusnum, 0.0)  # velociy along the cavity axis

        jx = 0.0
        jy = 0.0

        output = np.full(t_length // stpsize, 0.0)
        jxlist = np.full(t_length // stpsize, 0.0)
        jylist = np.full(t_length // stpsize, 0.0)

        nn = 0
        inoutindex = 0

        # for i in tqdm(range(t_length), desc='computing'):
        for i in range(t_length):
            # logging.info(f"{_STOPPED}")
            pbar.update(1)
            if _STOPPED == True:
                break
                time.sleep(1)
                logging.info("process stopped")
            tt = i * dt
            accumulated_phase += sqrtFWHM * np.random.normal(loc=0.0, scale=np.sqrt(dt))  # simulate phase noise
            jx = np.sum(eta * sx)
            jy = np.sum(eta * sy)

            if i % stpsize == 0:
                output[i // stpsize] = (np.abs(jx) ** 2.0 + np.abs(jy) ** 2.0)
                jxlist[i // stpsize] = jx
                jylist[i // stpsize] = jy

            ### stochstic variables
            xip = np.random.normal(0, np.sqrt(dt))
            xiq = np.random.normal(0, np.sqrt(dt))

            ########## Runge-Kutta 4th order #####################
            # Thomas Gard, Introduction to Stochastic Differential Equations p.201
            F0x = + Gammac / 2 * eta * (jx * sz - eta * sx * (sz + 1)) - GammaD / 2 * eta * (
                    jy * sz - eta * sy * (sz + 1))
            G0xp = - Gammac / np.sqrt(Gamma0) * eta * sz
            G0xq = - GammaD / np.sqrt(Gamma0) * eta * sz
            F0y = + Gammac / 2 * eta * (jy * sz - eta * sy * (sz + 1)) + GammaD / 2 * eta * (
                    jx * sz - eta * sx * (sz + 1))
            G0yp = - GammaD / np.sqrt(Gamma0) * eta * sz
            G0yq = + Gammac / np.sqrt(Gamma0) * eta * sz
            F0z = - Gammac / 2 * eta * (jx * sx + jy * sy - eta * (sx ** 2.0 + sy ** 2.0)) + GammaD / 2 * eta * (
                    jy * sx - jx * sy) - Gammac * eta ** 2.0 * (sz + 1)
            G0zp = + Gammac / np.sqrt(Gamma0) * eta * sx + GammaD / np.sqrt(Gamma0) * eta * sy
            G0zq = - Gammac / np.sqrt(Gamma0) * eta * sy + GammaD / np.sqrt(Gamma0) * eta * sx

            sx1 = sx + 0.5 * dt * F0x + 0.5 * (G0xp * xip + G0xq * xiq)
            sy1 = sy + 0.5 * dt * F0y + 0.5 * (G0yp * xip + G0yp * xiq)
            sz1 = sz + 0.5 * dt * F0z + 0.5 * (G0zp * xip + G0zq * xiq)

            eta1 = np.cos(z + 0.5 * dt * vz)
            jx1 = np.sum(eta1 * sx1)
            jy1 = np.sum(eta1 * sy1)

            F1x = + Gammac / 2 * eta1 * (jx1 * sz1 - eta1 * sx1 * (sz1 + 1)) - GammaD / 2 * eta1 * (
                    jy1 * sz1 - eta1 * sy1 * (sz1 + 1))
            G1xp = - Gammac / np.sqrt(Gamma0) * eta1 * sz1
            G1xq = - GammaD / np.sqrt(Gamma0) * eta1 * sz1
            F1y = + Gammac / 2 * eta1 * (jy1 * sz1 - eta1 * sy1 * (sz1 + 1)) + GammaD / 2 * eta1 * (
                    jx1 * sz1 - eta1 * sx1 * (sz1 + 1))
            G1yp = - GammaD / np.sqrt(Gamma0) * eta1 * sz1
            G1yq = + Gammac / np.sqrt(Gamma0) * eta1 * sz1
            F1z = - Gammac / 2 * eta1 * (
                    jx1 * sx1 + jy1 * sy1 - eta1 * (sx1 ** 2.0 + sy1 ** 2.0)) + GammaD / 2 * eta1 * (
                          jy1 * sx1 - jx1 * sy1) - Gammac * eta1 ** 2.0 * (sz1 + 1)
            G1zp = + Gammac / np.sqrt(Gamma0) * eta1 * sx1 + GammaD / np.sqrt(Gamma0) * eta1 * sy1
            G1zq = - Gammac / np.sqrt(Gamma0) * eta1 * sy1 + GammaD / np.sqrt(Gamma0) * eta1 * sx1

            sx2 = sx + 0.5 * dt * F1x + 0.5 * (G1xp * xip + G1xq * xiq)
            sy2 = sy + 0.5 * dt * F1y + 0.5 * (G1yp * xip + G1yp * xiq)
            sz2 = sz + 0.5 * dt * F1z + 0.5 * (G1zp * xip + G1zq * xiq)

            eta2 = eta1
            jx2 = np.sum(eta2 * sx2)
            jy2 = np.sum(eta2 * sy2)

            F2x = + Gammac / 2 * eta2 * (jx2 * sz2 - eta2 * sx2 * (sz2 + 1)) - GammaD / 2 * eta2 * (
                    jy2 * sz2 - eta2 * sy2 * (sz2 + 1))
            G2xp = - Gammac / np.sqrt(Gamma0) * eta2 * sz2
            G2xq = - GammaD / np.sqrt(Gamma0) * eta2 * sz2
            F2y = + Gammac / 2 * eta2 * (jy2 * sz2 - eta2 * sy2 * (sz2 + 1)) + GammaD / 2 * eta2 * (
                    jx2 * sz2 - eta2 * sx2 * (sz2 + 1))
            G2yp = - GammaD / np.sqrt(Gamma0) * eta2 * sz2
            G2yq = + Gammac / np.sqrt(Gamma0) * eta2 * sz2
            F2z = - Gammac / 2 * eta2 * (
                    jx2 * sx2 + jy2 * sy2 - eta2 * (sx2 ** 2.0 + sy2 ** 2.0)) + GammaD / 2 * eta2 * (
                          jy2 * sx2 - jx2 * sy2) - Gammac * eta2 ** 2.0 * (sz2 + 1)
            G2zp = + Gammac / np.sqrt(Gamma0) * eta2 * sx2 + GammaD / np.sqrt(Gamma0) * eta2 * sy2
            G2zq = - Gammac / np.sqrt(Gamma0) * eta2 * sy2 + GammaD / np.sqrt(Gamma0) * eta2 * sx2

            sx3 = sx + dt * F2x + G2xp * xip + G2xq * xiq
            sy3 = sy + dt * F2y + G2yp * xip + G2yp * xiq
            sz3 = sz + dt * F2z + G2zp * xip + G2zq * xiq

            eta3 = np.cos(z + dt * vz)
            jx3 = np.sum(eta3 * sx3)
            jy3 = np.sum(eta3 * sy3)

            F3x = + Gammac / 2 * eta3 * (jx3 * sz3 - eta3 * sx3 * (sz3 + 1)) - GammaD / 2 * eta3 * (
                    jy3 * sz3 - eta3 * sy3 * (sz3 + 1))
            G3xp = - Gammac / np.sqrt(Gamma0) * eta3 * sz3
            G3xq = - GammaD / np.sqrt(Gamma0) * eta3 * sz3
            F3y = + Gammac / 2 * eta3 * (jy3 * sz3 - eta3 * sy3 * (sz3 + 1)) + GammaD / 2 * eta3 * (
                    jx3 * sz3 - eta3 * sx3 * (sz3 + 1))
            G3yp = - GammaD / np.sqrt(Gamma0) * eta3 * sz3
            G3yq = + Gammac / np.sqrt(Gamma0) * eta3 * sz3
            F3z = - Gammac / 2 * eta3 * (
                    jx3 * sx3 + jy3 * sy3 - eta3 * (sx3 ** 2.0 + sy3 ** 2.0)) + GammaD / 2 * eta3 * (
                          jy3 * sx3 - jx3 * sy3) - Gammac * eta3 ** 2.0 * (sz3 + 1)
            G3zp = + Gammac / np.sqrt(Gamma0) * eta3 * sx3 + GammaD / np.sqrt(Gamma0) * eta3 * sy3
            G3zq = - Gammac / np.sqrt(Gamma0) * eta3 * sy3 + GammaD / np.sqrt(Gamma0) * eta3 * sx3

            sx += (F0x + 2 * F1x + 2 * F2x + F3x) * dt / 6 + (G0xp + 2 * G1xp + 2 * G2xp + G3xp) * xip / 6 + (
                    G0xq + 2 * G1xq + 2 * G2xq + G3xq) * xiq / 6
            sy += (F0y + 2 * F1y + 2 * F2y + F3y) * dt / 6 + (G0yp + 2 * G1yp + 2 * G2yp + G3yp) * xip / 6 + (
                    G0yq + 2 * G1yq + 2 * G2yq + G3yq) * xiq / 6
            sz += (F0z + 2 * F1z + 2 * F2z + F3z) * dt / 6 + (G0zp + 2 * G1zp + 2 * G2zp + G3zp) * xip / 6 + (
                    G0zq + 2 * G1zq + 2 * G2zq + G3zq) * xiq / 6
            z += dt * vz
            eta = np.cos(z)
            #############################################################

            while tt - nn * tau / clusnum > 0:  # atom in/out
                ## single atom injection
                # sx[inoutindex] = 1.0 if random.random() < (1 + np.sin(theta)*np.cos(-delpa*tt + accumulated_phase))/2.0 else -1.0 # projection noise
                # sy[inoutindex] = 1.0 if random.random() < (1 - np.sin(theta)*np.sin(-delpa*tt + accumulated_phase))/2.0 else -1.0
                # sz[inoutindex] = 1.0 if random.random() < rhoee else -1.0
                ## cluster injection
                sx[inoutindex] = 2.0 * np.sum(
                    np.random.random(cs) < (1 + np.sin(theta) * np.cos(-delpa * tt + accumulated_phase)) / 2.0) - cs
                sy[inoutindex] = 2.0 * np.sum(
                    np.random.random(cs) < (1 - np.sin(theta) * np.sin(-delpa * tt + accumulated_phase)) / 2.0) - cs
                sz[inoutindex] = 2.0 * np.sum(np.random.random(cs) < rhoee) - cs
                ## when atom num in a cluster is large
                # p1 = (1 + np.sin(theta)*np.cos(-delpa*tt + accumulated_phase))/2.0  ## prob that sx=1
                # mu = (2*p1-1)
                # sigma = np.sqrt(cs**2 + 2*mu*cs**2 + mu**2*cs**2 + 4*cs*p1 - 4*cs**2*p1 - 4*mu*cs**2*p1 - 4*cs*p1**2 +  4*cs**2*p1**2)
                # sx[inoutindex] = np.random.normal(cs*mu,sigma)
                # p1 = (1 - np.sin(theta)*np.sin(-delpa*tt + accumulated_phase))/2.0  ## prob that sy=1
                # mu = (2*p1-1)
                # sigma = np.sqrt(cs**2 + 2*mu*cs**2 + mu**2*cs**2 + 4*cs*p1 - 4*cs**2*p1 - 4*mu*cs**2*p1 - 4*cs*p1**2 +  4*cs**2*p1**2)
                # sy[inoutindex] = np.random.normal(cs*mu,sigma)
                # p1 = rhoee                                                                 ## prob that sz=1
                # mu = (2*p1-1)
                # sigma = np.sqrt(cs**2 + 2*mu*cs**2 + mu**2*cs**2 + 4*cs*p1 - 4*cs**2*p1 - 4*mu*cs**2*p1 - 4*cs*p1**2 +  4*cs**2*p1**2)
                # sz[inoutindex] = np.random.normal(cs*mu,sigma)

                if randomphase == 'on':
                    z[inoutindex] = random.random() * 2.0 * np.pi  # injection location (random)
                else:
                    z[inoutindex] = 0.0  # injection location (anti-node only)
                vz[inoutindex] = np.random.normal(delT, delD)  # velocity along the cavity axis rad/us
                nn += 1
                inoutindex = nn % clusnum
            if i % 100 == 0:
                _PROGRESS_STATUS.value += 1
                # logging.info(f"{_PROGRESS_STATUS.value}")
                # WorkerSignals.signal.emit(100, t_length)
        output = Gammac * tau * output / 4 / clusnum / cs  # photon per atom

        return output, jxlist, jylist


class Atomicbeamclock(QThread):
    result_signal = pyqtSignal(list, np.ndarray)
    kappatau = pyqtSignal(float)
    kappagsqrt = pyqtSignal(float)
    kappadelD = pyqtSignal(float)
    NtauGammac = pyqtSignal(float)
    unittime = pyqtSignal(float)
    longtimeerror = pyqtSignal(str)
    finished = pyqtSignal()
    stop_finished = pyqtSignal()
    progress_signal = pyqtSignal(int)
    test = 1

    def __init__(self, ntraj=1000, atomtype="Ba-138", cs=2, clusnum=1000, kappa=2 * np.pi * 230, g=2 * np.pi * 0.22,
                 tau=0.14,
                 ctlth=10000, stpsize=8, cstpsiz=1, delDtau=0.2 * np.pi, delT=2 * np.pi * 0.0, delca=2 * np.pi * 2.5,
                 delpa=2 * np.pi * 1.2, rhoee=0.9, pumplinewidth=2 * np.pi * 0.01, manipulated_var='clusnum',
                 dependent_var='outputpower', randomphase='on', fftcalc='on',
                 fftplot='on', fftfit='on'):
        super().__init__()
        self.ntraj = ntraj
        self.atomtype = atomtype
        self.cs = cs
        self.clusnum = clusnum
        self.clusnum_list = np.linspace(5, self.clusnum, 20).astype(int)
        self.kappa = kappa
        self.g = g
        self.tau = tau
        self.ctlth = ctlth
        self.stpsize = stpsize
        self.cstpsiz = cstpsiz
        self.dt = self.tau / 200
        self.delDtau = delDtau
        self.delD = self.delDtau / self.tau
        self.delT = delT
        self.delca = delca
        self.delpa = delpa
        self.dca_list = np.linspace(-2 * self.delca, 0.0, 50)
        self.dpa_list = np.linspace(-self.delpa, self.delpa, 50)
        self.rhoee = rhoee
        self.theta = 2.0 * np.arcsin(np.sqrt(self.rhoee))
        self.pumplinewidth = pumplinewidth
        self.sqrtFWHM = np.sqrt(self.pumplinewidth)
        self.accumulated_phase = 0.0
        self.manipulated_var = manipulated_var
        self.dependent_var = dependent_var
        self.randomphase = randomphase
        self.fftcalc = fftcalc
        self.fftplot = fftplot
        self.fftfit = fftfit
        self.Gammac = self.g ** 2.0 * self.kappa / 4.0 / (self.kappa ** 2.0 / 4.0 + self.delca ** 2.0)
        self.Gamma0 = self.g ** 2.0 / self.kappa
        self.GammaD = self.g ** 2.0 * self.delca / 2.0 / (self.kappa ** 2.0 / 4.0 + self.delca ** 2.0)
        self.pool = Pool(processes=multiprocessing.cpu_count() - 2)
        self.forced = False
        # self.q = Queue()
        global _STOPPED

    def normlrtz(x, amp1, cen1, wid1):  # wid1: FWHM
        return amp1 / np.pi * (wid1 / 2.0) / ((x - cen1) ** 2.0 + (wid1 / 2.0) ** 2.0)

    def run(self):
        cs, clusnum, kappa, g, tau, ctlth, stpsize, cstpsiz, dt, delDtau, delD, delT, delca, delpa, rhoee, \
        pumplinewidth, sqrtFWHM, accumulated_phase, manipulated_var, dependent_var = self.cs, self.clusnum, self.kappa, self.g, \
                                                                                     self.tau, self.ctlth, self.stpsize, \
                                                                                     self.cstpsiz, self.dt, self.delDtau, \
                                                                                     self.delD, self.delT, self.delca, self.delpa, \
                                                                                     self.rhoee, self.pumplinewidth, self.sqrtFWHM, \
                                                                                     self.accumulated_phase, self.manipulated_var, \
                                                                                     self.dependent_var
        randomphase = self.randomphase
        fftcalc = self.fftcalc
        fftplot = self.fftplot
        fftfit = self.fftfit
        theta = self.theta


        if dependent_var == 'evolve':
            self.t_final = 5 * tau  # microsec, simulation time
            self.stpsize = 1
        elif dependent_var == 'outputpower':
            self.t_final = 500 * tau  # microsec, simulation time
            self.stpsize = 1
        else:
            self.t_final = 10000  # microsec, simulation time

        self.t_length = int(self.t_final / dt)
        self.t_list = np.linspace(0, self.t_final, self.t_length)

        if manipulated_var == 'clusnum':
            vlist = cs * self.clusnum_list
        elif manipulated_var == 'dca':
            vlist = self.dca_list
        elif manipulated_var == 'dpa':
            vlist = self.dpa_list

        logging.info("process start")
        numofproc = multiprocessing.cpu_count() - 2
        _STOPPED = False
        self.timer = QTimer(self)
        self.timer.start(1000)
        self.timer.timeout.connect(self.timeout)
        result = self.pool.map(Sngltraj_pool(*self.params()).run,
                               np.arange(len(self.clusnum_list)))  # starmap  : for multiple argumetns

        if not self.forced:
            self.forced = False
            self.pool.close()
            self.pool.join()
            self.finished.emit()

        self.result_signal.emit(result, vlist)
        # return result, vlist

    def timeout(self):
        self.progress_signal.emit(_PROGRESS_STATUS.value)

    def params(self):
        return self.ntraj, self.atomtype, self.cs, self.clusnum, self.kappa, self.g, self.tau, self.ctlth, self.stpsize, self.cstpsiz, self.dt,\
               self.delDtau, self.delD, self.delT, self.delca, self.delpa, self.rhoee, self.pumplinewidth, \
               self.sqrtFWHM, self.accumulated_phase, self.manipulated_var, self.dependent_var, self.randomphase, \
               self.fftcalc, self.fftplot, self.fftfit, self.theta

    def show_params(self):
        print("ntraj=", self.ntraj, ", manipulated_var=", self.manipulated_var, ", dependent_var=", self.dependent_var,
              ", randomphase=", self.randomphase, ", FFT calc=", self.fftcalc,
              ", FFT plot=", self.fftplot, ", FFT fit=", self.fftfit, ", tau=", self.tau, ", atomtype=", self.atomtype)

    def stop(self):
        self.forced = True
        logging.info("stopping")
        _STOPPED = True
        # print("_STOPPED = ", _STOPPED)
        self.pool.close()
        self.pool.join()
        time.sleep(2)
        self.stop_finished.emit()

    def analysis(self, result, vlist, args):
        ntraj, atomtype, cs, clusnum, kappa, g, tau, ctlth, stpsize, cstpsiz, dt, delDtau, delD, delT, delca, delpa, rhoee, \
        pumplinewidth, sqrtFWHM, accumulated_phase, manipulated_var, dependent_var, randomphase, fftcalc, fftplot, fftfit, theta = args

        if self.dependent_var == 'evolve':
            output = []
            for i in range(ntraj):
                output.append(result[i][0])
            # output,jxlist,jylist = sngltraj(kappa,g,delca,delpa,tau,clusnum,accumulated_phase,sqrtFWHM)
            output = sum(np.array(output)) / ntraj
            print(np.average(output[len(output) // 2:]))
            ######################################################

            ### plot and sav the results
            fig = plt.figure()
            plt.plot(self.t_list, output, '.')
            fig.savefig('evolve_' + str(rhoee) + '_atomn_' + str(clusnum) + '_delT_' + str(
                delT / 2.0 / np.pi) + '_deltDtau_' + str(delDtau / np.pi) + '_dca_' + str(
                delca / np.pi / 2) + '_dpa_' + str(delpa / np.pi / 2) + '_rp_' + randomphase + '_plwth_' + str(
                pumplinewidth / np.pi / 2) + '_.png')
            f = open('evolve' + str(rhoee) + '_atomn_' + str(clusnum) + '_delT_' + str(
                delT / 2.0 / np.pi) + '_deltDtau_' + str(delDtau / np.pi) + '_dca_' + str(
                delca / np.pi / 2) + '_dpa_' + str(delpa / np.pi / 2) + '_rp_' + randomphase + '_plwth_' + str(
                pumplinewidth / np.pi / 2) + '.txt', 'w')
            np.savetxt(f, np.vstack((self.t_list, output)).T)  # ,newline=' ')
            f.close()

        if self.dependent_var == "outputpower":
            output = []
            for i in range(len(vlist)):
                output.append(result[i][0])
            output = np.array(output)
            p_avg = []
            for i in range(len(vlist)):
                p_avg.append(np.average(output[i][len(output):]))
            ######################################################

            fig = plt.figure()
            if manipulated_var == 'clusnum':
                vlist = vlist * self.Gammac * tau
            else:
                vlist = vlist / 2 / np.pi
            plt.plot(vlist, p_avg, '.')
            plt.ylim([0, 1])
            fig.savefig('output_rhoee_' + str(rhoee) + '_atomn_' + str(clusnum) + '_delT_' + str(
                delT / 2.0 / np.pi) + '_deltDtau_' + str(delDtau / np.pi) + '_dca_' + str(
                delca / np.pi / 2) + '_dpa_' + str(delpa / np.pi / 2) + '_rp_' + randomphase + '.png')
            f = open('output_rhoee_' + str(rhoee) + '_atomn_' + str(clusnum) + '_delT_' + str(
                delT / 2.0 / np.pi) + '_deltDtau_' + str(delDtau / np.pi) + '_dca_' + str(
                delca / np.pi / 2) + '_dpa_' + str(delpa / np.pi / 2) + '_rp_' + randomphase + '.txt', 'w')
            np.savetxt(f, np.vstack((vlist, p_avg)).T)  # ,newline=' ') #*Gammac*tau
            f.close()

        if self.dependent_var == "g1calc":
            jxx = []
            jyy = []
            for i in range(len(vlist)):
                jxx.append(result[i][1])
                jyy.append(result[i][2])
            jxx = np.array(jxx)
            jyy = np.array(jyy)

            now = time.localtime()
            print('g1calc start')
            print("%04d/%02d/%02d %02d:%02d:%02d" % (
            now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec))
            fig_g1c = plt.figure()
            fig_fft = plt.figure()
            fig_lnw = plt.figure()
            ax1 = fig_g1c.subplots()
            ax2 = fig_fft.subplots()
            ax3 = fig_lnw.subplots()
            vp_list = []  # xaxis
            amp_list = []
            cen_list = []  # lasing frequency
            sigma_list = []  # FWHM

            # Create a new directory if it does not exist
            path = os.getcwd() + '//g1sav'
            isExist = os.path.exists(path)
            if not isExist:
                os.makedirs(path)
            for i in prange(len(vlist)):
                if manipulated_var == 'clusnum':
                    vary = vlist[i] * self.Gammac * tau
                else:
                    vary = vlist[i] / 2 / np.pi
                # jxlist = jxx[i][len(jxx[0])//10:] # steadystate only
                # jylist = jyy[i][len(jyy[0])//10:] # steadystate only
                jxlist = jxx[i][1:]  # steadystate only
                jylist = jyy[i][1:]  # steadystate only
                ########### g1 calculation #############
                cftn = []  # auto correlation fuction
                jxplist = jxlist[:-cstpsiz * ctlth]
                jyplist = jylist[:-cstpsiz * ctlth]
                for index in range(ctlth):
                    jxpplist = jxlist[cstpsiz * index:cstpsiz * index - cstpsiz * ctlth]
                    jypplist = jylist[cstpsiz * index:cstpsiz * index - cstpsiz * ctlth]
                    # cftn.append((np.dot(jxplist,jxpplist))/len(jxplist)+(np.dot(jyplist,jypplist))/len(jyplist))
                    cftn.append((np.dot(jxpplist + 1.0j * jypplist, jxplist - 1.0j * jyplist)) / len(jxplist) / 4.0)

                cftn = np.array(cftn)
                # cftn  = cftn/max(cftn)                          # normalization g^(1)(0)=1
                ctime = cstpsiz * stpsize * dt * np.array(range(ctlth))  # x axis for g1 ftn
                cftn *= np.exp(1.0j * delT * ctime)  # rotating frame

                ax1.plot(ctime, cftn, '.')
                ax1.plot(ctime, cftn)

                ###### save the result
                fig_g1c.savefig('g1sav_' + str(rhoee) + '_atomn_' + str(clusnum) + '_delT_' + str(
                    delT / 2.0 / np.pi) + '_deltDtau_' + str(delDtau / np.pi) + '_dca_' + str(
                    delca / np.pi / 2) + '_dpa_' + str(delpa / np.pi / 2) + '_rp_' + randomphase + '_plwth_' + str(
                    pumplinewidth / np.pi / 2) + '.png')
                f = open('g1sav/g1sav_rhoee_' + str(rhoee) + '_atomn_' + str(clusnum) + '_delT_' + str(
                    delT / 2.0 / np.pi) + '_deltDtau_' + str(delDtau / np.pi) + '_dca_' + str(
                    delca / np.pi / 2) + '_dpa_' + str(delpa / np.pi / 2) + '_rp_' + randomphase + '_plwth_' + str(
                    pumplinewidth / np.pi / 2) + '_v_' + str(vary) + '.txt', 'w')
                np.savetxt(f, np.vstack((np.real(ctime), np.real(cftn), np.imag(cftn))).T)
                f.close()

                if fftcalc == 'on':
                    x = ctime
                    y = cftn
                    x = np.concatenate((-x[::-1][:-1], x))
                    y = np.concatenate((np.conjugate(y[::-1][:-1]), y))
                    g_x = x
                    g_y = y
                    yf = fftpack.fft(y, x.size)
                    amp = np.abs(yf)  # get amplitude spectrum
                    freq = fftpack.fftfreq(x.size, (x[1] - x[0]))  # MHz
                    ind = freq.argsort()
                    amp = amp[ind]
                    freq = freq[ind]
                    f = open('g1sav/fftsav_rhoee_' + str(rhoee) + '_atomn_' + str(clusnum) + '_delT_' + str(
                        delT / 2.0 / np.pi) + '_deltDtau_' + str(delDtau / np.pi) + '_dca_' + str(
                        delca / np.pi / 2) + '_dpa_' + str(delpa / np.pi / 2) + '_rp_' + randomphase + '_plwth_' + str(
                        pumplinewidth / np.pi / 2) + '_v_' + str(vary) + '.txt', 'w')
                    np.savetxt(f, np.vstack(
                        (np.real(freq), (1 / amp.size) * np.real(np.abs(amp) / (freq[1] - freq[0])))).T)
                    f.close()
                    y = (1 / amp.size) * amp / (freq[1] - freq[0])
                    if fftplot == 'on':
                        ax2.plot(freq, (1 / amp.size) * amp / (freq[1] - freq[0]), 'k.', label=vlist[i])
                        ax2.plot(freq, (1 / amp.size) * amp / (freq[1] - freq[0]), 'k-', alpha=0.8)
                        # ax2.set_xlim(-3.0,3.0)
                    # plt.yscale('log')
                    # plt.ylim(bottom=0.0)
                    # legend = plt.legend(loc='upper right')
                    if fftfit == 'on':
                        try:
                            index = np.argmax(np.abs(y))
                            fitrange = 30
                            yfit = np.abs(y[index - fitrange:index + fitrange])
                            xfit = np.real(freq[index - fitrange:index + fitrange])
                            initial_guess = [max(yfit), np.real(freq[index]),
                                             (max(xfit) - min(xfit)) / 2]  # p0=[amp1, cen1, sigma1]
                            popt, pcov = curve_fit(self.normlrtz, xfit, yfit, p0=initial_guess)
                            # print("FWHM_lasing")
                            # print(popt[-1])
                            if manipulated_var == 'clusnum':
                                vp_list.append(vlist[i] * self.Gammac * tau)
                            else:
                                vp_list.append(vlist[i] / 2 / np.pi)

                            amp_list.append(popt[0])
                            cen_list.append(popt[1])
                            sigma_list.append(popt[2])
                            if fftplot == 'on':
                                ax2.plot(xfit, self.normlrtz(xfit, *popt), color='red', linewidth=2, label="fitting")
                                ax2.set_xlim(xfit[0], xfit[-1])
                                # legend = plt.legend(loc='upper right')
                        except:
                            print("fftfitting error occured!")
            if fftplot == 'on':
                fig_fft.savefig('fftsav_' + str(rhoee) + '_atomn_' + str(clusnum) + '_delT_' + str(
                    delT / 2.0 / np.pi) + '_deltDtau_' + str(delDtau / np.pi) + '_dca_' + str(
                    delca / np.pi / 2) + '_dpa_' + str(delpa / np.pi / 2) + '_rp_' + randomphase + '_plwth_' + str(
                    pumplinewidth / np.pi / 2) + '.png')
            if fftfit == 'on':
                ax3.plot(vp_list, sigma_list)
                ax3.set_yscale('log')
                fig_lnw.savefig('fftresult_' + str(rhoee) + '_atomn_' + str(clusnum) + '_delT_' + str(
                    delT / 2.0 / np.pi) + '_deltDtau_' + str(delDtau / np.pi) + '_dca_' + str(
                    delca / np.pi / 2) + '_dpa_' + str(delpa / np.pi / 2) + '_rp_' + randomphase + '_plwth_' + str(
                    pumplinewidth / np.pi / 2) + '.png')
                f = open('fftresult_rhoee_' + str(rhoee) + '_atomn_' + str(clusnum) + '_delT_' + str(
                    delT / 2.0 / np.pi) + '_deltDtau_' + str(delDtau / np.pi) + '_dca_' + str(
                    delca / np.pi / 2) + '_dpa_' + str(delpa / np.pi / 2) + '_rp_' + randomphase + '_plwth_' + str(
                    pumplinewidth / np.pi / 2) + '.txt', 'w')
                np.savetxt(f, np.vstack((np.real(vp_list), amp_list, cen_list, sigma_list)).T)
                f.close()

if __name__ == "__main__":
    # ntraj1 = 1000
    # cs = 1
    # manipulated_var1 = "clusnum"
    # dependent_var1 = "outputpower"
    # randomphase1 = 'on'
    # fftcalc1 = 'off'
    # fftplot1 = 'off'
    # fftfit1 = 'off'
    # test = Atomicbeamclock(ntraj=1)
    # test.show_params()
    # # print(test.pbar)
    # test.run()
    # # test.run()
    trajcompute = Atomicbeamclock()
    # Sngltraj_pool(*trajcompute.params()).run()
    result = trajcompute.run()
    trajcompute.analysis(result, vlist, trajcompute.params())