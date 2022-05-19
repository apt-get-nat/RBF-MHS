function Bn = rbf_mhs(nodes,k,m,d,B0,dens,pres,index_0,Bpot,varargin)
%rbf_mhs Wrapper function that builds rbf matrices and calls num_mhs.
% This is provided more as an example than meant to be called repeatedly.
% In particular, it performs the RBF precomputation as part of its routine,
% and repeated computations on the same nodeset would do well to separate
% that from the num_mhs call.
%   nodes    -- nx3 node locations in space
%   k        -- number of nearest neighbors in rbf stencil (eg, 95)
%   m        -- degree of polyharmonic spline (eg, 5)
%   d        -- degree of added polynomials (eg, 4)
%   B0       -- (3*n0)x1 Dirischlet condition
%   dens     -- nx1 density prescribed at node locations
%   pres     -- nx1 pressure prescribed at node locations
%   index_0  -- locations in nodes of the Dirischlet boundary condition
%   Bpot     -- 3*nx1 initial guess and preconditioning field

n = size(nodes,1);
n0 = size(index_0,1);

% Physical unit scalar
g = 1.6e-12;

% Build the RBF matrix discretization
% You may find in practice that varying k over the domain (eg, more near
% the boundaries) will provide better results than this rudimentary
% process
idx = knnsearch(nodes,nodes,'k',k);
w = zeros(n,k, 9);
for k = 1:n
    xx = nodes(idxfull(k,:),1);
    yy = nodes(idxfull(k,:),2);
    zz = nodes(idxfull(k,:),3);
    w( k,:,:) = RBF_FD_PHS_pol_weights_3D (xx,yy,zz,d,m);
end
Dx = sparse(repmat(idx(:,1),1,k),idx,w(:,:,1),n,n);
Dy = sparse(repmat(idx(:,1),1,k),idx,w(:,:,2),n,n);
Dz = sparse(repmat(idx(:,1),1,k),idx,w(:,:,3),n,n);
Dxx = sparse(repmat(idx(:,1),1,k),idx,w(:,:,4),n,n);
Dyy = sparse(repmat(idx(:,1),1,k),idx,w(:,:,6),n,n);
Dzz = sparse(repmat(idx(:,1),1,k),idx,w(:,:,9),n,n);

Bn = num_mhs(dens,pres,B0(1:n),B0(n+1:2*n),B0(2*n+1:3*n), n, ...
        nodes, Dx, Dy, Dz, Dxx, Dyy, Dzz, g,...
        index_x0,index_x1,index_y0,index_y1,index_z0,index_z1,index_gh, ...
        1e-4, Bpot);

end