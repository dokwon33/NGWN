import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np

data = np.log10(np.array(self.getCellUsers()).transpose() + 1)

fig = plt.figure(figsize=(12, 6))
ax = plt.axes(projection=ccrs.PlateCarree())

# 데이터 overlay
img = ax.imshow(
    data,
    extent=[-180, 180, -90, 90],  # 중요!!
    transform=ccrs.PlateCarree(),
    cmap='viridis',
    origin='lower'
)

# 지도 요소 추가
ax.coastlines(linewidth=0.8)
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.set_global()

plt.colorbar(img, ax=ax, label='log10(User Count)')
plt.title('Global User Distribution')
plt.show()