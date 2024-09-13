from setuptools import setup, find_packages

setup(
    name='machine_learning',  # Nome do pacote
    version='0.1.0',  # Versão do pacote
    author='Seu Nome',  # Nome do autor
    author_email='seu.email@example.com',  # E-mail do autor
    description='Descrição do seu projeto',  # Breve descrição do projeto
    long_description=open('README.md').read(),  # Descrição longa (geralmente o conteúdo de README.md)
    long_description_content_type='text/markdown',  # Tipo do conteúdo da descrição longa
    url='https://github.com/seuusuario/machine_learning',  # URL do repositório do projeto
    packages=find_packages(),  # Inclui todos os pacotes encontrados no diretório
    classifiers=[  # Classificadores que ajudam a categorizar o pacote
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',  # Versão mínima do Python requerida
    install_requires=[  # Dependências do projeto
        'numpy',
        'pandas',
        'scikit-learn',
    ],
    extras_require={  # Dependências opcionais
        'dev': ['pytest', 'sphinx'],
    },
    entry_points={  # Entradas de ponto para a execução de scripts
        'console_scripts': [
            'my_command=machine_learning.module:function',
        ],
    },
)
