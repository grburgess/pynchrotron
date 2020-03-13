from setuptools import setup
import versioneer


setup(

    version=versioneer.get_version(),
    license="GPL",
    cmdclass=versioneer.get_cmdclass()
)


