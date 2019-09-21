import numpy as np
from pyproj import Proj
import matplotlib.pyplot as plt
# from mpl_toolkits.basemap import pyproj
# import seaborn
# from mpl_toolkits.basemap import Basemap
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation

class SinglePlanarFault(object):
  def __init__(self,
               top_center,
               dims,
               angles,
               ngrid,
               hyp,
               vr,
               dt,
               mu,
               sigma,
               dmax,
               rake,
               t_acc,
               t_eff
               ):
    p = Proj(proj='utm', zone=34, ellps='WGS84',preserve_units=False)
    self._p = p
    self._elon = top_center[1]
    self._elat = top_center[0]
    self._lth = dims[0]
    self._dth = top_center[2]
    self._wid = dims[1]
    self._stk = angles[0]
    self._dip = angles[1]
    self._nstk = ngrid[0]
    self._ndip = ngrid[1]
    self._shyp = hyp[0]
    self._dhyp = hyp[1]
    self._dt = dt
    self._vr = vr
    lth = self._lth * 1e3
    wid = self._wid * 1e3
    dth = self._dth * 1e3
    stk = self._stk * np.pi / 180.0
    dip = self._dip * np.pi / 180.0
    pts = self._nstk * self._ndip
    self._area = lth * wid / pts * 1e4
    pwid = wid * np.cos(dip)
    dl = lth / self._nstk
    dw = pwid / self._ndip
    alon = []
    alat = []
    cd = 0.
    x,y =  p(self._elon, self._elat)
    txl = []
    tyl = []
    tx, ty = x - lth/2.0* np.cos(stk),y + lth/2.0 * np.cos(np.pi/2. - stk)
    for i in range(self._nstk * 2):
      tx = tx + 0.5 * dl * np.cos(stk)
      ty = ty - 0.5 * dl * np.cos(np.pi/2. - stk)
      if i % 2 ==0:
        txl.append(tx)
        tyl.append(ty)
    pxl = []
    pyl = []
    dep = []
    d = dth
    px, py = x,y
    for j in range(self._ndip * 2):
      px = px + 0.5 * dw * np.cos(np.pi/2. + stk)
      py = py - 0.5 * dw * np.sin(np.pi/2. + stk)
      d = d + 0.5 * wid * np.sin(dip) / self._ndip
      if j % 2 ==0:
        pxl.append(px)
        pyl.append(py)
        dep.append(d)
    self._c = np.zeros((self._nstk, self._ndip, 3))
    self._tinit = np.zeros((self._nstk, self._ndip))
    for i in range(self._nstk):
      for j in range(self._ndip):
        tmpx, tmpy = txl[i] + pxl[j] - x, tyl[i] + pyl[j] - y
        nlon, nlat = p(tmpx, tmpy, inverse=True)
        self._c[i,j,0] = nlon
        self._c[i,j,1] = nlat
        self._c[i,j,2] = dep[j]
    self._hlon, self._hlat, self._hdep = self._c[hyp[0], hyp[1], 0], \
      self._c[hyp[0], hyp[1], 1], self._c[hyp[0], hyp[1], 2]

    hx, hy = p(self._hlon, self._hlat)
    for i in range(self._nstk):
      for j in range(self._ndip):
        self._tinit[i,j] = np.sqrt((txl[i] + pxl[j] - x-hx)**2 +
                                   (tyl[i] + pyl[j] - y - hy)**2)/(self._vr * 1e3)

    self._model_to_fault(mu, sigma, dmax, rake, t_acc, t_eff)
    self._rake = []
    self._slip1 = []
    self._slip2 = []
    self._slip3 = []
    self._sr1 = []
    self._sr2 = []
    self._sr3 = []
    for i in range(self._nstk):
      for j in range(self._ndip):
        # k = i * self._ndip + j
        self._rake.append(self._rake_mat[i, j])
        self._slip1.append(self._slip_mat[i, j])
        self._slip2.append(0.0)
        self._slip3.append(0.0)
        self._sr1.append(list(self._slip_vel_mat[i,j,:]))
        self._sr2.append([0.0]*200)
        self._sr3.append([0.0]*200)
    # self._parse_bm_file(vel_model)
    # print(self._hypdist)

  # def _parse_bm_file(self, filename='model.bm'):
  #   f = open(filename, 'r')
  #   lines = f.readlines()[5::]
  #   # print(lines)
  #   d = []
  #   vs = []
  #   rho = []
  #   c = 0
  #   for i, l in enumerate(lines):
  #     if l.find('#'):
  #       if i==0:
  #         radius = float(l.split()[0])
  #       if i % 2 == c:
  #         l_split = l.split()
  #         d.append(radius - float(l_split[0]))
  #         vs.append(float(l_split[3]))
  #         rho.append(float(l_split[1]))
  #     else:
  #       c = (c+1) % 2
  #   f.close()
  #   self._model_d = np.array(d)
  #   self._model_vs = np.array(vs)
  #   self._model_rho = np.array(rho)
  #   print(self._model_d)

  def _model_to_fault(self, mu, sigma, dmax, rake, t_acc, t_eff):
    n = self._nstk * self._ndip
    self._slip_mat = np.zeros((self._nstk, self._ndip))
    self._t_acc_mat = np.zeros((self._nstk, self._ndip))
    self._t_eff_mat = np.zeros((self._nstk, self._ndip))
    self._rake_mat = np.zeros((self._nstk, self._ndip))
    self._slip_vel_mat = np.zeros((self._nstk, self._ndip, 200))
    for i in range(self._nstk):
      for j in range(self._ndip):
        self._slip_mat[i,j] = self._estimate_slip((i,j), dmax, mu, sigma)
        self._t_acc_mat[i,j] = self._estimate_q((i,j), dmax, t_acc, mu, sigma)
        self._t_eff_mat[i,j] = self._estimate_q((i,j), dmax, t_eff, mu, sigma)
        self._rake_mat[i,j] = self._estimate_q((i,j), dmax, rake, mu, sigma)
    for i in range(self._nstk):
      for j in range(self._ndip):
        self._slip_vel_mat[i,j,:] = self._slip_velocity((i,j))

  def _estimate_q(self, idx, q0, q1, mu, sigma):
    k = len(mu)
    sum1 = 0.0
    sum2 = 0.0
    x, y, z = self._idx_to_xyz_km(idx)
    for i in range(k):
      _mu_x, _mu_y, _mu_z = self._idx_to_xyz_km(mu[i])
      d = self._dist_km(_mu_x, _mu_y, _mu_z, x, y, z)
      sum1 += q0[i] * q1[i] * self._gaussian_max_one(d, sigma[i])
      sum2 += q0[i] * self._gaussian_max_one(d, sigma[i])
    return sum1/sum2

  def _estimate_slip(self, idx, q, mu, sigma):
    k = len(mu)
    sum1 = 0.0
    sum2 = 0.0
    x, y, z = self._idx_to_xyz_km(idx)
    for i in range(k):
      _mu_x, _mu_y, _mu_z = self._idx_to_xyz_km(mu[i])
      d = self._dist_km(_mu_x, _mu_y, _mu_z, x, y, z)
      sum1 += q[i] * self._gaussian_max_one(d, sigma[i])
    return sum1


  def _gaussian_max_one(self, d, s):
    nu = np.exp(-0.5 * d**2 / s**2)
    return nu

  def _idx_to_xyz_km(self, idx):
    nlon, nlat, z = self._c[idx[0], idx[1], 0], self._c[idx[0], idx[1], 1], self._c[idx[0], idx[1], 2]
    x, y =  self._p(nlon, nlat)
    return x/1000, y/1000, z/1000

  def _dist_km(self, x1, y1, z1, x2, y2, z2):
    return np.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)

  def animation_slip_velocity(self, filename, num_steps=1000):
    # num_steps = 600
    t = 0.0
    ts = np.zeros_like((self._slip_mat)) - 1
    inst = []
    times = []
    for epoch in range(num_steps):
      c = self._tinit < t
      ts[c] += 1
      ts = np.minimum(ts, 199 * np.ones_like((self._slip_mat)))
      times.append(t)
      t += self._dt
      tmp = np.zeros_like(ts)
      for i in range(self._nstk):
        for j in range(self._ndip):
          tmp[i,j] = self._slip_vel_mat[i,j,int(ts[i,j])]
      inst.append(tmp)
    fig = plt.figure()
    plt.gca().invert_yaxis()
    x = np.linspace(-self._lth/2, self._lth/2, self._nstk)
    y = np.linspace(0.0, self._wid, self._ndip)
    plt.xlabel('distance along strike [km]')
    plt.ylabel('distance along dip [km]')
    ims = []
    for i in range(num_steps):
      ims.append((plt.pcolor(x,y,np.transpose(inst[i]), vmin=0.0, vmax=10.0),))
    im_ani = animation.ArtistAnimation(fig, ims, interval=10, repeat_delay=300, blit=True)
    im_ani.save(filename)
    # plt.show()


  def _slip_velocity(self, idx):
    t1 = 2.0
    t = np.arange(0, t1, self._dt)
    slip_vel = np.zeros((t.shape[0],))
    for i in range(t.shape[0]):
      slip_vel[i] = self._slip_velocity_t(t[i],
                                          self._slip_mat[idx[0], idx[1]],
                                          self._t_acc_mat[idx[0], idx[1]],
                                          self._t_eff_mat[idx[0], idx[1]])
    return slip_vel



  def _slip_velocity_t(self, t, dmax, t_acc, t_eff):
    tr = t_eff - 2. * t_acc / 1.27
    ts = t_acc/1.27
    k = 2. / (np.pi * tr * ts**2)

    c1 = (0.5 * t + 0.25 * tr) * np.sqrt(t * (tr - t)) + \
      (t*tr - tr**2) * np.arcsin(np.sqrt(t/tr)) - \
      0.75 * tr**2 * np.arctan(np.sqrt((tr - t) / t))

    c2 = 3. * np.pi * tr**2 / 8.

    c3 = (ts - t - 0.5 * tr) * np.sqrt((t - ts) * (tr - t + ts)) + \
      tr * (2. * tr - 2. * t + 2. * ts) * np.arcsin(np.sqrt((t - ts) / tr)) + \
      1.5 * tr**2 * np.arctan(np.sqrt((tr - t + ts) / (t - ts)))

    c4 = (-ts + 0.5 * t + 0.25 * tr) * np.sqrt((t - 2. * ts) * (tr - t + 2. * ts)) + \
      tr * (- tr + t - 2. * ts) * np.arcsin(np.sqrt((t - 2. * ts) / tr)) - \
      0.75 * tr**2 * np.arctan(np.sqrt((tr - t + 2. * ts) / (t - 2. * ts)))

    c5 = 0.5 * np.pi * tr * (t - tr)

    c6 = 0.5 * np.pi * tr * (2. * ts - t + tr)

    if t < 0:
      s = 0
    elif t < ts:
      s = c1 + c2
    elif tr > 2. * ts and t < 2. * ts:
      s = c1 - c2 + c3
    elif tr <= 2. * ts and t < tr:
      s = c1 - c2 + c3
    elif tr > 2. * ts and t < tr:
      s = c1 + c3 + c4
    elif tr <= 2. * ts and t < 2. * ts:
      s = c5 + c3 - c2
    elif tr > 2. * ts and t < tr + ts:
      s = c5 + c3 + c4
    elif tr <= 2. * ts and t < tr + ts:
      s = c5 + c3 + c4
    elif t < tr + 2. * ts:
      s = c4 + c6
    elif t >= tr + 2. * ts:
      s = 0
    return s * k * dmax

  def create_srf(self, filename='/home/kesmarag/single_planar_fault.srf'):
    f = open(filename, 'w+')
    # version
    # f.write('%.1f\n'% 2.0)
    # Plane
    f.write('PLANE 1\n')
    f.write('  %.4f %.4f %d %d %.2f %.2f\n'\
            %(self._elon, self._elat, self._nstk, self._ndip,
              self._lth, self._wid))
    f.write('  %d %d %.2f %.2f %.2f\n'\
            %(self._stk, self._dip, self._dth, self._shyp, self._dhyp))
    # Points
    f.write('POINTS %d\n' %(self._ndip * self._nstk))
    for i in range(self._nstk):
      for j in range(self._ndip):
        # k = next(x[0] for x in enumerate(self._model_d) if x[1] > self._c[i,j,2])
        # print(self._c[i,j,2])
        # print('k =', k-1)
        # vs = self._model_vs[k-1]
        # rho = self._model_rho[k-1]
        # f.write('  %.4f %.4f %.4f %d %d %.5e %.4f %.5e %.5e %.5e\n'
        f.write('  %.4f %.4f %.4f %d %d %.5e %.4f %.5e\n'
                %(self._c[i,j,0], self._c[i,j,1], self._c[i,j,2]/1000,
                  self._stk, self._dip, self._area,
                  self._tinit[i,j], self._dt))
                  # self._tinit[i,j], self._dt, vs*100, rho/1000))
        q = i * self._ndip + j
        if self._slip2[q] == 0.0:
          tmp2 = 0
        else:
          tmp2 = len(self._sr2[q])
        if self._slip3[q] == 0.0:
          tmp3 = 0
        else:
          tmp3 = len(self._sr3[q])
        f.write('  %d %.2f %d %.2f %d %.2f %d\n'\
                %(self._rake[q],self._slip1[q],len(self._sr1[q]),self._slip2[q],tmp2,self._slip3[q],tmp3))
        for m in range(len(self._sr1[q])):
          f.write('  %.5e'% self._sr1[q][m])
        f.write('\n')
        for m in range(tmp2):
          f.write('  %.5e'% self._sr2[q][m])
          if m==tmp2-1:
            f.write('\n')
        for m in range(tmp3):
          f.write('  %.5e'% self._sr3[q][m])
          if m==tmp3-1:
            f.write('\n')
    f.close()
  # def plot(self):
  #   fig, ax = plt.subplots(figsize=(12, 12))
  #   m = Basemap(
  #               # projection='merc',
  #               urcrnrlat=40., llcrnrlat=36.,
  #               urcrnrlon=23., llcrnrlon=19.,
  #               resolution='l',
  #               suppress_ticks=False,
  #             ax=ax)
  #   m.fillcontinents()
  #   m.drawcoastlines()
  #   m.ax = ax
  #   for i in range(self._nstk):
  #     for j in range(self._ndip):
  #       plt.plot(self._c[i,j,0], self._c[i,j,1], '*')
  #   plt.plot(self._elon, self._elat, 'o')
  #   plt.plot(self._hlon, self._hlat, 'o')
  #   plt.show()

if __name__ == '__main__':
  angles = (30, 40)
  ngrid = (40, 30)
  dims = (16.0, 12.0)
  top_center = (38.0, 20.0, 3.0)
  # hyp = (-5.0, 5.0)
  hyp_idx = (10, 10)
  vr = 2.6
  vel_model = 'wgmf.bm'
  dt = 0.01
  mu = [(8, 5), (13, 14), (28, 22), (20, 20), (16, 7), (17, 4)]
  sigma = [0.9, 1.5, 1.0, 0.4, 1.3, 1.2]
  # hyp_idx = (0, 1)
  dmax = [4.0, 7.0, 10.0, 3.0, 8.0, 2.0]
  rake = [50.0, 40.0, 50.0, 40.0, 40.0, 40.0]
  t_acc = [0.2, 0.1, 0.3, 0.1, 0.2, 0.1]
  t_eff = [1.2, 1.5, 1.2, 1.4, 1.1, 1.4]


  fault = SinglePlanarFault(top_center,
                            dims,
                            angles,
                            ngrid,
                            hyp_idx,
                            vr,
                            dt,
                            mu,
                            sigma,
                            dmax,
                            rake,
                            t_acc,
                            t_eff)
  fault.create_srf()
  # fault.plot()
  # idx_test = (1,1)
  # mu_test = [(0,1), (1,1)]
  # sigma_test = [10.0, 5.0]
  # q = [1.0, 10.0]
  # qstar = fault._estimate_q(idx_test, q, mu_test, sigma_test)
  # print(qstar)
  # v = []
  # T = np.arange(-0.5, 2.0, 0.005)
  # for t in T:
  #   v.append(fault._slip_velocity_t(t, 3.0 , 0.15, 0.75))
  # fault._model_to_fault(mu, sigma, dmax, rake, t_acc, t_eff)
  # fault.animation_slip_velocity('/home/kesmarag/animation.mp4')
  # vel = fault._slip_velocity((20,2))
  # plt.plot(vel)
  # plt.pcolor(fault._rake_mat)
  # plt.colorbar()
  # plt.plot(fault._slip_vel_mat[1,1,:])
  # plt.show()
  # fault._gaussian_max_one(x_test, mu_test, sigma_test)
  # x1, y1, z1 = fault._idx_to_xyz_km((1,1))
  # x2, y2, z2 = fault._idx_to_xyz_km((1,0))
  # d = fault._dist_km(x1,y1,z1,x2,y2,z2)
  # print(d)
  # fault.create_srf()
  # print(v)
  # plt.plot(T, v)
  # plt.pcolor(fault._tinit)
  # plt.show()
  # fault.plot()
  # print(fault._tinit)
