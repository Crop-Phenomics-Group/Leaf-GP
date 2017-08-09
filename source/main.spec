# -*- mode: python -*-
# this is the spec file used by PyInstaller to create executable

block_cipher = None

a = Analysis(['./main.py'],
             pathex=['./'],
             binaries=[],
             datas=[('no_image.png', '.'), ('information_icon.png', '.'), ('information_icon_disabled.png', '.'), ('tick.png', '.'), ('untick.png', '.')],
             hiddenimports=['pywt._extensions._cwt', 'sklearn.neighbors.typedefs'],
             # make sure this points to the root directory of the project where hook-analysis.py is. May need to put absolute path if not loaded
             hookspath=['./'],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)

exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          name='Leaf-GP',
          debug=False,
          strip=False,
          upx=True,
          console=False )

coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               name='main')
app = BUNDLE(coll,
             name='Leaf-GP.app',
             version=0.9,
             bundle_identifier=None)
