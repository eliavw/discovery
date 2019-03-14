# Deployment Information

For use on a new machine, we use conda. On the original machine, we always work
inside an isolated python environment, managed by conda. 

This environment can be exported to a `.yml` file through the following command:

`conda env export > environment.yml`

Which creates the `.yml` file present in the root dir. 

To recreate this environment, it suffices to run;

`conda env create -f environment.yml -n <whatever name you want>`

Which presupposes that you have an anaconda install running on your own machine.
In theory, this should be portable enough.
