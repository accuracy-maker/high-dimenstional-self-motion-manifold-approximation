# high-dimenstional-self-motion-manifold-approximation
codebase for my SMM paper

## Run command:
```shell
chmod +x run.sh
./run.sh all
```

## Results
### 3R
![3R planar motion](evaluation/3R_planar.png)

### 7R Panda | pose
![7R panda pose](evaluation/franka_emika_panda_pose.png)

### 7R iiwa | pose
![7R iiwa pose](evaluation/kuka_iiwa_14_pose.png)

you can see that all samples from flow-matching model lie on the correct SMMs computed by ODE method. It's a subset of 1-D SMM curves as it's contrainted by **joint limits**
