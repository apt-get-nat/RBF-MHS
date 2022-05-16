% MHS_driver_rbf.m
% Author: Nathaniel H. Mathews
% -------------------------------------------------------------------------
% This code is provided as an example to call the num_mhs function as well
% as others, and so construct a numerical magnetohydrostatic (MHS) magnetic
% field, which aligns with the true analytic solution constructed by Gibson
% & Low (1998).

clear all;

% RBF methods often result in high condition number, but the condition
% number of those matrices turns out to not be the good holistic for
% approximate singularity that it usually is.
warning('off','MATLAB:nearlySingularMatrix');

% For scaling or rescaling the physical equations
physics.mp = 1;
physics.kB = 1;
physics.k_m = 1;
physics.m_k = 1;
physics.g = 1.6e-12;


% Next we construct the scattered nodeset we will use in the simulation.
Ltyp = 7;

% nside = 15; hexn = 15;
% nside = 30; hexn = 27;
nside = 60; hexn = 53;

omega = 2; % exponential stretching

hexnodes = HexLayout(hexn)'*2-1;
nslice = size(hexnodes,1);

if omega ~= 0
    zside = linspace(0,log(Ltyp+1)/omega,nside);
    zfn = @(node) exp(omega*node(:,3))-1;
else
    zside = linspace(0,Ltyp,nside);
    zfn = @(node) node(:,3);
end
xfn = @(node) ( node(:,1) )*Ltyp/2;
yfn = @(node) ( node(:,2) )*Ltyp/2;
           
% An inclusion of a ghost layer will be important for the number of
% boundary conditions we impose later
zside = [zside(end:-1:1),-zside(2)];

nodes = repmat(hexnodes,nside+1,1);
nodes(:,3) = reshape(repmat(zside,nslice,1),[],1);

n = size(nodes,1);

%% Next we get the RBF discritization weights for the domain.
% This occurs in two parts. First, we find the nearest neighbors (the IDX),
% and then we pass them in to the RBF-FD code to generate weights. Here in
% both cases we will use the symmetry of our particular nodeset across
% hexagonally-structured planes, but the most general case would just be a
% single knnsearch() call and then n distinct RBF_FD calls.

% We choose rbfk and panels to result in 155 total nearest neighbors per
% interior node. This is a number compatible with the 4th order polynomials
% we are using. Higher accuracy would require higher-order polynomials, and
% thus more nearest neighbors (as per pascal's triangle).
rbfk = 19;
rbfkbd = 31;
panels = 5;
rbfp = 4;

% find boundary nodes
panelbd = [];
for j = 1:4
    panelbd = [panelbd;find( hexnodes(:,1) == min(hexnodes(setdiff(1:nslice,panelbd),1)) )];
    panelbd = [panelbd;find( hexnodes(:,1) == max(hexnodes(setdiff(1:nslice,panelbd),1)) )];
end
for j = 1:2
    panelbd = [panelbd;find( hexnodes(:,2) == min(hexnodes(setdiff(1:nslice,panelbd),2)) )];
    panelbd = [panelbd;find( hexnodes(:,2) == max(hexnodes(setdiff(1:nslice,panelbd),2)) )];
end
panelbd = unique(panelbd);
panelint = setdiff(1:nslice,panelbd);
nbd = numel(panelbd);
nint = numel(panelint);

% find nearest neighbors within a panel
idx = knnsearch(hexnodes,hexnodes(panelint,:),'k',rbfk);
idxbd = knnsearch(hexnodes,hexnodes(panelbd,:),'k',rbfkbd);

% use indexing to populate the full 3D nearest neighbor tree
idxfull = zeros(nint*numel(zside),rbfk*panels);
idxbdfull = zeros(nbd*numel(zside),rbfkbd*panels);
w = zeros(nint*numel(zside), rbfk*panels, 9);
wbd = zeros(nbd*numel(zside), rbfkbd*panels, 9);
for j = 1:numel(zside) % for each panel
    fprintf('j = %i/%i\n',j,numel(zside));
    
    % compute 3d nearest neighbors
    if j <= (panels-1)/2 % top boundary handling
        if j == 1, offset = (0:1:panels-1)'; end
        if j == 2, offset = [1,0,2:panels-1]'; end
    elseif j > numel(zside)-(panels-1)/2 % bottom boundary handling
        if j == numel(zside), offset = numel(zside)+(-1:-1:-panels)'; end
        if j == numel(zside)-1, offset = numel(zside)+[-2,-1,-3:-1:-panels]'; end
    else % interior with respect to z
        offset = [j-1,j-1-(panels-1)/2:1:j-2,j:1:j+(panels-1)/2-1]';
    end
    offsetbd = (offset.*ones(panels,rbfkbd)*nslice)'; offsetbd = offsetbd(:)';
    offset = (offset.*ones(panels,rbfk)*nslice)'; offset = offset(:)';
    
    idxfull( (j-1)*nint+1:j*nint, : ) = repmat(idx,1,5) + offset;
    idxbdfull( (j-1)*nbd+1:j*nbd, : ) = repmat(idxbd,1,5) + offsetbd;
    
    % compute RBF weights
    if j > (panels-1)/2+1 && j < numel(zside)-(panels-1)/2
        w( (j-1)*nint+1:j*nint,:,:) = w( (j-2)*nint+1:(j-1)*nint,:,:);
        wbd( (j-1)*nbd+1:j*nbd,:,:) = wbd( (j-2)*nbd+1:(j-1)*nbd,:,:);
    else
        for k = 1:nbd % for node within panel
            xx = nodes(idxbdfull((j-1)*nbd+k,:),1);
            yy = nodes(idxbdfull((j-1)*nbd+k,:),2);
            zz = nodes(idxbdfull((j-1)*nbd+k,:),3);
            wbd( (j-1)*nbd+k,:,:) = RBF_FD_PHS_pol_weights_3D (xx,yy,zz,5,rbfp);
        end
        for k = 1:nint
            xx = nodes(idxfull((j-1)*nint+k,:),1);
            yy = nodes(idxfull((j-1)*nint+k,:),2);
            zz = nodes(idxfull((j-1)*nint+k,:),3);
            w( (j-1)*nint+k,:,:) = RBF_FD_PHS_pol_weights_3D (xx,yy,zz,5,rbfp);
        end
    end
    
end

Dx0 = sparse(repmat(idxfull(:,1),1,rbfk*panels),idxfull,w(:,:,1),n,n) + ...
      sparse(repmat(idxbdfull(:,1),1,rbfkbd*panels),idxbdfull,wbd(:,:,1),n,n);
Dy0 = sparse(repmat(idxfull(:,1),1,rbfk*panels),idxfull,w(:,:,2),n,n) + ...
      sparse(repmat(idxbdfull(:,1),1,rbfkbd*panels),idxbdfull,wbd(:,:,2),n,n);
Dz0 = sparse(repmat(idxfull(:,1),1,rbfk*panels),idxfull,w(:,:,3),n,n) + ...
      sparse(repmat(idxbdfull(:,1),1,rbfkbd*panels),idxbdfull,wbd(:,:,3),n,n);
Dxx0 = sparse(repmat(idxfull(:,1),1,rbfk*panels),idxfull,w(:,:,4),n,n) + ...
       sparse(repmat(idxbdfull(:,1),1,rbfkbd*panels),idxbdfull,wbd(:,:,4),n,n);
Dyy0 = sparse(repmat(idxfull(:,1),1,rbfk*panels),idxfull,w(:,:,6),n,n) + ...
       sparse(repmat(idxbdfull(:,1),1,rbfkbd*panels),idxbdfull,wbd(:,:,6),n,n);
Dzz0 = sparse(repmat(idxfull(:,1),1,rbfk*panels),idxfull,w(:,:,9),n,n) + ...
       sparse(repmat(idxbdfull(:,1),1,rbfkbd*panels),idxbdfull,wbd(:,:,9),n,n);

% rescale the Dz and Dzz matrices. We could just include these factors when
% we use the matrices later, but this consolidates the transformation.
Dz = Dz0./(omega*exp(omega*nodes(:,3)));
Dzz = Dzz0./(omega*exp(2*omega*nodes(:,3))) - Dz./(omega*exp(omega*nodes(:,3)));

% Similarly, we could have computed the RBF calculation on xfn(nodes) and
% yfn(nodes), but the problem is better regularized if we scale to the
% physical domain after.
Dx = Dx0*2/Ltyp;
Dy = Dy0*2/Ltyp;
Dxx = Dxx0*(2/Ltyp)^2;
Dyy = Dyy0*(2/Ltyp)^2;

% Finally, we store the indices of various boundary nodes for later use.
index_x0 = find(nodes(:,1) == min(nodes(:,1)));
index_x1 = find(nodes(:,1) == max(nodes(:,1)));
index_y0 = find(nodes(:,2) == min(nodes(:,2)));
index_y1 = find(nodes(:,2) == max(nodes(:,2)));
index_z0 = find(nodes(:,3) == 0);
index_z1 = find(nodes(:,3) == max(nodes(:,3)));
index_gh = find(nodes(:,3) == min(nodes(:,3)));
index_in = setdiff(1:n,[index_x0',index_x1',...
        index_y0',index_y1',index_z0',index_z1',index_gh'])';


%% Retrieve analytic true solution
% Of course, such a solution is not in general necessary and may not be
% available. The only important aspect for the numerical solver is that
% boundary conditions are provided. We take those boundary conditions from
% the analytic solution.

% We provide in this repository the analytic gibson-low solution for depth
% 0.9 (arc-shaped) and depth=0.825 (doughnut-shaped) fields at the
% following resolutions: 15 x 15, 27 x 30 and 53 x 60.
depth = 0.9;
fname = sprintf('.\\giblow_nodes\\L7nodes_%i_%i_omega%i_%.6f.sav',hexn,nside,omega,depth);
data = restore_idl(fname);

% the data as provided has By oriented in the vertical direction.
By = -data.BX';
Bz = data.BY';
Bx = data.BZ';
dens = data.DENS';
pres = data.PRES';
temp = data.TEMP';
clear data;
    
Bt = [Bx;By;Bz;dens;pres];

% boundary conditions
Bx0   =   Bx(index_z0);
By0   =   By(index_z0);
Bz0   =   Bz(index_z0);

%% Construct initial guess and preconditioning field
% The potential field model turns out to be a poor approximation of the
% magnetic field we are considering; we use instead open up-and-down field
% lines divided across the y-axis.

Bpx = zeros(n,1);
Bpy = zeros(n,1);
Bpz = zeros(n,1);
Bpz(nodes(:,1)>0) = 0.1;
Bpz(nodes(:,1)<0) = -0.1;
Bpot = [Bpx;Bpy;Bpz];
switch 1
    case 0
        
    case 1
        load(sprintf('L7preconfun_giblow_depth%.6f_eachiter.mat',depth));
        Bpx = Bpxfun(xfn(nodes),yfn(nodes),zfn(nodes));
        Bpy = Bpyfun(xfn(nodes),yfn(nodes),zfn(nodes));
        Bpz = Bpzfun(xfn(nodes),yfn(nodes),zfn(nodes));
end
        

%% Numerical solver
% That's it for setup; now we can actually run the model.

% numerical hyperviscocity -- may need to be tweaked depending on the
% problem. Use 0 for no hyperviscocity (results may be unstable!)
gamma = 1e-4; tic;

% This is the core call we've spent 200+ lines of code building up to:
[Bn,rs,Bp_unmodified,Bfullset] = num_mhs(dens,pres,Bx0,By0,Bz0, n, ...
        nodes, Dx, Dy, Dz, Dxx, Dyy, Dzz, physics.g,...
        index_x0,index_x1,index_y0,index_y1,index_z0,index_z1,index_gh, ...
        gamma, [Bpx;Bpy;Bpz], Bpot);
toc

% Visualizations
% To present magnetic field lines and slice plots. Because the data is
% scattered, we need to reinterpolate it onto a regular grid to use these
% MATLAB plotting commands.

numX = scatteredInterpolant([xfn(nodes),yfn(nodes),zfn(nodes)],Bn(1:n),'nearest');
numY = scatteredInterpolant([xfn(nodes),yfn(nodes),zfn(nodes)],Bn(n+1:2*n),'nearest');
numZ = scatteredInterpolant([xfn(nodes),yfn(nodes),zfn(nodes)],Bn(2*n+1:3*n),'nearest');

truX = scatteredInterpolant([xfn(nodes),yfn(nodes),zfn(nodes)],Bt(1:n),'nearest');
truY = scatteredInterpolant([xfn(nodes),yfn(nodes),zfn(nodes)],Bt(n+1:2*n),'nearest');
truZ = scatteredInterpolant([xfn(nodes),yfn(nodes),zfn(nodes)],Bt(2*n+1:3*n),'nearest');

truDens = scatteredInterpolant([xfn(nodes),yfn(nodes),zfn(nodes)],dens(1:n),'nearest');
truPres = scatteredInterpolant([xfn(nodes),yfn(nodes),zfn(nodes)],pres(1:n),'nearest');

step = 0.0625;
[xq,yq,zq] = meshgrid((-Ltyp/2:step:Ltyp/2),(-Ltyp/2:step:Ltyp/2),0:step/2:Ltyp/2);
Bxnq = numX(xq,yq,zq);
Bynq = numY(xq,yq,zq);
Bznq = numZ(xq,yq,zq);
Bxtq = truX(xq,yq,zq);
Bytq = truY(xq,yq,zq);
Bztq = truZ(xq,yq,zq);
densq = truDens(xq,yq,zq);
presq = truPres(xq,yq,zq);

% this is the stage where we save the output so we don't have to run
% everything again.
save(sprintf('savedata\\L7n%i_giblow_depth%.6f_omega%i_neverclean.mat',nside,depth,omega),'-v7.3');
fprintf('Saved.\n');

%% Plot streamlines
close all
step = 4;

startx = squeeze(xq(1:step:end,1:step:end,1));
starty = squeeze(yq(1:step:end,1:step:end,1));
startz = squeeze(zq(1:step:end,1:step:end,1));
startpts = [startx(:),starty(:),startz(:)];
clear startx starty startz

startpts = startpts(startpts(:,1).^2 + startpts(:,2).^2 < 2,:);

fig = figure(11);
ax = axes('Parent',fig);
h1 = streamline(xq,yq,zq,Bxnq,Bynq,Bznq, ...
    startpts(:,1),startpts(:,2),startpts(:,3));
hold on;
h2 = streamline(xq,yq,zq,-Bxnq,-Bynq,-Bznq, ...
    startpts(:,1),startpts(:,2),startpts(:,3));
h3 = slice(xq,yq,zq,Bznq,[],[],0);
colormap('jet');
title(sprintf('Numeric field lines, $N\\approx %i^3$',nside),'interpreter','latex');
set(h1,'Color','k'); set(h2,'Color','k'); set(h3,'edgecolor','flat');
view(3); axis([-Ltyp/2 Ltyp/2 -Ltyp/2 Ltyp/2 0 Ltyp/2]);
set(gca,'Fontsize',16);
view(ax,[-1.11062148337596 23.7747787348907]);
xlabel('x'); ylabel('y'); zlabel('z');
hold off;

fig = figure(21);
ax = axes('Parent',fig);
h1 = streamline(xq,yq,zq,Bxtq,Bytq,Bztq, ...
    startpts(:,1),startpts(:,2),startpts(:,3));
hold on;
h2 = streamline(xq,yq,zq,-Bxtq,-Bytq,-Bztq, ...
    startpts(:,1),startpts(:,2),startpts(:,3));
h3 = slice(xq,yq,zq,Bztq,[],[],0);
colormap('jet');
title('True field lines','interpreter','latex');
set(h1,'Color','k'); set(h2,'Color','k'); set(h3,'edgecolor','flat');
view(3); axis([-Ltyp/2 Ltyp/2 -Ltyp/2 Ltyp/2 0 Ltyp/2]);
set(gca,'Fontsize',16);
view(ax,[-1.11062148337596 23.7747787348907]);
xlabel('x'); ylabel('y'); zlabel('z');
hold off;

%% Error plots

j = 2;

fig = figure(1);
ax = axes('Parent',fig);
h = slice(xq,yq,zq,abs(Bztq-Bznq),[],[],[j]); colorbar();
colormap('jet'); caxis([0 0.5]);
title(sprintf('$|B_z^{num}-B_z^{true}|$ at $z=%.1f$',j),'interpreter','latex');
set(h,'edgecolor','flat');
xlabel('x'); ylabel('y'); zlabel('z');
axis([-Ltyp/2 Ltyp/2 -Ltyp/2 Ltyp/2 0 Ltyp/2]); view(ax,[165.9375 14.4302088409328]);
set(gca,'FontSize',14);
view(2);

fig = figure(2);
ax = axes('Parent',fig);
h = slice(xq,yq,zq,Bztq,[],[],[j]); colorbar();
colormap('jet');
title(sprintf('$B_z^{true}$ at $z=%.1f$',j),'interpreter','latex');
set(h,'edgecolor','flat');
xlabel('x'); ylabel('y'); zlabel('z');
axis([-Ltyp/2 Ltyp/2 -Ltyp/2 Ltyp/2 0 Ltyp/2]); view(ax,[165.9375 14.4302088409328]);
set(gca,'FontSize',14);
view(2);

fig = figure(3);
ax = axes('Parent',fig);
h = slice(xq,yq,zq,abs(Bxtq-Bxnq),[],[],[j]); colorbar();
colormap('jet'); caxis([0 0.5]);
title(sprintf('$|B_x^{num}-B_x^{true}|$ at $z=%.1f$',j),'interpreter','latex');
set(h,'edgecolor','flat');
xlabel('x'); ylabel('y'); zlabel('z');
axis([-Ltyp/2 Ltyp/2 -Ltyp/2 Ltyp/2 0 Ltyp/2]); view(ax,[165.9375 14.4302088409328]);
set(gca,'FontSize',14);
view(2);

fig = figure(4);
ax = axes('Parent',fig);
h = slice(xq,yq,zq,Bxtq,[],[],[j]); colorbar();
colormap('jet');
title(sprintf('$B_x^{true}$ at $z=%.1f$',j),'interpreter','latex');
set(h,'edgecolor','flat');
xlabel('x'); ylabel('y'); zlabel('z');
axis([-Ltyp/2 Ltyp/2 -Ltyp/2 Ltyp/2 0 Ltyp/2]); view(ax,[165.9375 14.4302088409328]);
set(gca,'FontSize',14);
view(2);

fig = figure(5);
ax = axes('Parent',fig);
h = slice(xq,yq,zq,abs(Bytq-Bynq),[],[],[j]); colorbar();
colormap('jet'); caxis([0 0.5]);
title(sprintf('$|B_x^{num}-B_x^{true}|$ at $z=%.1f$',j),'interpreter','latex');
set(h,'edgecolor','flat');
xlabel('x'); ylabel('y'); zlabel('z');
axis([-Ltyp/2 Ltyp/2 -Ltyp/2 Ltyp/2 0 Ltyp/2]); view(ax,[165.9375 14.4302088409328]);
set(gca,'FontSize',14);
view(2);

fig = figure(6);
ax = axes('Parent',fig);
h = slice(xq,yq,zq,Bytq,[],[],[j]); colorbar();
colormap('jet');
title(sprintf('$B_y^{true}$ at $z=%.1f$',j),'interpreter','latex');
set(h,'edgecolor','flat');
xlabel('x'); ylabel('y'); zlabel('z');
axis([-Ltyp/2 Ltyp/2 -Ltyp/2 Ltyp/2 0 Ltyp/2]); view(ax,[165.9375 14.4302088409328]);
set(gca,'FontSize',14);
view(2);

%% Slice plots
xslice = [];
yslice = [];
zslice = [0,1,2,3];

fig = figure(1);
ax = axes('Parent',fig);
h = slice(xq,yq,zq,Bznq,xslice,yslice,zslice); colorbar();
colormap('jet');
title('Numeric Bz','interpreter','latex');
set(h,'edgecolor','flat');
xlabel('x'); ylabel('y'); zlabel('z');
axis([-Ltyp/2 Ltyp/2 -Ltyp/2 Ltyp/2 0 3]); view(ax,[165.9375 14.4302088409328]);
set(gca,'FontSize',14);

fig = figure(2);
ax = axes('Parent',fig);
h = slice(xq,yq,zq,Bztq,xslice,yslice,zslice); colorbar();
colormap('jet');
title('True Bz','interpreter','latex');
set(h,'edgecolor','flat');
xlabel('x'); ylabel('y'); zlabel('z');
axis([-Ltyp/2 Ltyp/2 -Ltyp/2 Ltyp/2 0 3]); view(ax,[165.9375 14.4302088409328]);
set(gca,'FontSize',14);

fig = figure(3);
ax = axes('Parent',fig);
h = slice(xq,yq,zq,abs(Bztq-Bznq),xslice,yslice,zslice); colorbar();
colormap('jet');
title('Error in Bz','interpreter','latex');
set(h,'edgecolor','flat');
xlabel('x'); ylabel('y'); zlabel('z');
axis([-Ltyp/2 Ltyp/2 -Ltyp/2 Ltyp/2 0 3]); view(ax,[165.9375 14.4302088409328]);
set(gca,'FontSize',14);
%%
fig = figure(4);
ax = axes('Parent',fig);
h = slice(xq,yq,zq,abs(Bxtq-Bxnq),xslice,yslice,zslice); colorbar();
colormap('jet');
title('Error in Bx');
set(h,'edgecolor','flat');
xlabel('x'); ylabel('y'); zlabel('z');
axis([-Ltyp/2 Ltyp/2 -Ltyp/2 Ltyp/2 0 Ltyp/2]); view(ax,[165.9375 14.4302088409328]);
set(gca,'FontSize',14);

fig = figure(5);
ax = axes('Parent',fig);
h = slice(xq,yq,zq,abs(Bytq-Bynq),xslice,yslice,zslice); colorbar();
colormap('jet');
title('Error in By');
set(h,'edgecolor','flat');
xlabel('x'); ylabel('y'); zlabel('z');
axis([-Ltyp/2 Ltyp/2 -Ltyp/2 Ltyp/2 0 Ltyp/2]); view(ax,[165.9375 14.4302088409328]);
set(gca,'FontSize',14);

fig = figure(6);
ax = axes('Parent',fig);
h = slice(xq,yq,zq,densq,xslice,yslice,zslice); colorbar();
colormap('jet');
title('True density');
set(h,'edgecolor','flat');
xlabel('x'); ylabel('y'); zlabel('z');
axis([-Ltyp/2 Ltyp/2 -Ltyp/2 Ltyp/2 0 Ltyp/2]); view(ax,[165.9375 14.4302088409328]);
set(gca,'FontSize',14);

fig = figure(7);
ax = axes('Parent',fig);
h = slice(xq,yq,zq,presq,xslice,yslice,zslice); colorbar();
colormap('jet');
title('True pressure');
set(h,'edgecolor','flat');
xlabel('x'); ylabel('y'); zlabel('z');
axis([-Ltyp/2 Ltyp/2 -Ltyp/2 Ltyp/2 0 Ltyp/2]); view(ax,[165.9375 14.4302088409328]);
set(gca,'FontSize',14);

%%
err = sqrt((Bxtq-Bxnq).^2+(Bytq-Bynq).^2+(Bztq-Bznq).^2);% ...
        %/max(sqrt(Bxtq(:).^2+Bytq(:).^2+Bztq(:).^2));

for j = [0,0.5,1,1.5,2,2.5,3,3.5]
    fig = figure(2*j+1);
    ax = axes('Parent',fig);
    h = slice(xq,yq,zq,abs(Bztq-Bznq),[],[],[j]); colorbar();
    colormap('jet'); caxis([0 0.5]);
    title(sprintf('$|B_z-B_z^{true}|$ at $z=%.1f$',j),'interpreter','latex');
    set(h,'edgecolor','flat');
    xlabel('x'); ylabel('y'); zlabel('z');
    axis([-Ltyp/2 Ltyp/2 -Ltyp/2 Ltyp/2 0 Ltyp/2]); view(ax,[165.9375 14.4302088409328]);
    set(gca,'FontSize',14);
    view(2);

    fig = figure(2*j+10);
    ax = axes('Parent',fig);
    h = slice(xq,yq,zq,sqrt((Bxtq-Bxnq).^2+(Bytq-Bynq).^2+(Bztq-Bznq).^2),[],[],[j]); colorbar();
    colormap('jet'); caxis([0 0.5]);
    title(sprintf('$||B^{final}-B^{true}||_2$ at $z=%.1f$',j),'interpreter','latex');
    set(h,'edgecolor','flat');
    xlabel('x'); ylabel('y'); zlabel('z');
    axis([-Ltyp/2 Ltyp/2 -Ltyp/2 Ltyp/2 0 Ltyp/2]); view(ax,[165.9375 14.4302088409328]);
    set(gca,'FontSize',14);
    view(2);
end

%% div calc
numDiv = scatteredInterpolant([xfn(nodes),yfn(nodes),zfn(nodes)],[Dx,Dy,Dz]*Bn,'nearest');
divq = numDiv(xq,yq,zq);

%% div plot

fig = figure(5);
ax = axes('Parent',fig);
h = slice(xq,yq,zq,abs(divq),[],[],[0,1,2,3]); colorbar();
cmp = colormap('parula');
title('Magnitude of Numeric Divergence','interpreter','latex');
set(h,'edgecolor','flat');
xlabel('x'); ylabel('y'); zlabel('z');
axis([-Ltyp/2 Ltyp/2 -Ltyp/2 Ltyp/2 0 3]); view(ax,[165.9375 14.4302088409328]);
set(gca,'FontSize',14);
set(gca,'colorscale','log'); caxis([1e-6 1e-4]);


%% Save as preconditioning function

% Bpxfun = scatteredInterpolant([xfn(nodes),yfn(nodes),zfn(nodes)],Bn(1:n));
% Bpyfun = scatteredInterpolant([xfn(nodes),yfn(nodes),zfn(nodes)],Bn(n+1:2*n));
% Bpzfun = scatteredInterpolant([xfn(nodes),yfn(nodes),zfn(nodes)],Bn(2*n+1:3*n));
% save(sprintf('L7preconfun_giblow_depth%.6f_eachiter.mat',depth),'Bpxfun','Bpyfun','Bpzfun');