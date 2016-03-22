% This source code is written by Ninh Pham (ndap@itu.dk) as a part of the MADAMS project 
% Feel free to re-use and re-distribute this source code for any purpose, 
% And cite our work when you re-use this code.
%--------------------------------------------------------------------------
% Function:
% - Load data as SVM format (downloaded from SVM website)
% - Do random feature mapping (TS, RM)   
% - Save data as MATRIX format
%
% Note:
% - Default data filename: exp (SVM format)
% - Default data filename: exp0 (MATRIX format)
% - Default TS filename: exp1 (MATRIX format)
% - Default RM filename: exp2 (MATRIX format)
%
%--------------------------------------------------------------------------

function SVM_RF_MATRIX()

% clear;
% clc;

%--------------------------------------------------------------------------
% Initialization of data set
%--------------------------------------------------------------------------

PATH        = 'F:\Study\Dropbox\Working\_Code\Matlab\TensorSketch\Dataset\Processed\a9a\';

% Original data
FILE            = 'exp';
FILE            = strcat(PATH, FILE);

[LABEL SPARSE]  = libsvmread(FILE);
DATA            = full(SPARSE);

clear SPARSE
[N, D]          = size(DATA);   % N rows and D columns

% Saving matrix format data 
FILE   = 'exp0';
FILE   = strcat(PATH, FILE);
save(FILE, 'DATA', '-ASCII');


% Initialization of kernel function : K(x, y) = (dot(x, y) + C)^k
C       = 0;            
K       = 2;
CS_COL  = 1000; % Count Sketch dimensionality or Number of random features


% Add an extra dimension to all points
temp    = repmat(C, N, 1);
DATA    = [DATA temp];
D       = D + 1;


% TensorSketch
tic
tCS_DATA    = FFT_CountSketch_k_Naive(DATA, K, CS_COL);
toc

% Saving matrix format data
FILE   = 'exp1';
FILE   = strcat(PATH, FILE);
save(FILE, 'tCS_DATA', '-ASCII');


% RMFM
tic
mCS_DATA    = RMFM(DATA(:, 1 : D - 1), K, C, CS_COL);
toc

% Saving matrix format data
FILE   = 'exp2';
FILE   = strcat(PATH, FILE);
save(FILE, 'mCS_DATA', '-ASCII');


%--------------------------------------------------------------------------
% Apply k-Level Tensoring Count Sketch on DATA
%--------------------------------------------------------------------------
function DATA_SKETCH = FFT_CountSketch_k_Naive(DATA, K, CS_COL)

[N, D]      = size(DATA);

% Generate 2 Hash functions for N points
indexHASH   = randi(CS_COL, K, D);                  % Matrix of K x D
bitHASH     = double(randi(2, K, D) - 1.5 ) * 2;    % Matrix of K x D

%----------------------
% Count Sketch for DATA
%----------------------

DATA_SKETCH = zeros(N, CS_COL);                     % Matrix of N x CS_COL
P           = zeros(K, CS_COL);                     % Matrix of K x CS_COL

% Loop all points Xi
for Xi = 1 : N
    
    temp   = DATA(Xi, :);                         % Data Xi
    P      = zeros(K, CS_COL);                    % Matrix of K x CS_COL
    
    % Sketching each element Xij of Xi
    for Xij = 1 : D

        % For each polynomials
        for Ki = 1 : K

            iHashIndex          = indexHASH(Ki, Xij);
            iHashBit            = bitHASH(Ki, Xij);
            P(Ki, iHashIndex)   = P(Ki, iHashIndex) + iHashBit * temp(Xij);

        end           

    end

    % FFT conversion
    P = fft(P, [], 2);
    
    % Component-wise product
    temp = prod(P, 1);
    
    % iFFT conversion
    DATA_SKETCH(Xi, :) = ifft(temp);
    
end

clear indexHASH bitHASH



%--------------------------------------------------------------------------
% Apply Random Maclaurin Feature Maps on DATA
%--------------------------------------------------------------------------
function DATA_SKETCH = RMFM(DATA, K, C, CS_COL)

[N, D] = size(DATA);

% Fix a value p > 1
P = 2;

% Generate Maclaurin coefficients
COEF = zeros(1, K + 1);
for i = 1 : K + 1    
    COEF(i) = nchoosek(K, i - 1) * C^(K + 1 - i);
end

% Generate random features
DATA_SKETCH = zeros(N, CS_COL);

% Generate a non negative integer N with prob. P(N = n) = 1/p^(n+1)
R = floor(log2(1 ./ rand(1, CS_COL)));

for i = 1 : CS_COL
    
    % Choosing a random degree to approximate    
    iRan = R(i);
    
    if iRan > K
        continue;
    end
    
    % Constant coefficients of polynomial
    const = sqrt(COEF(iRan + 1)) * sqrt(P^(iRan + 1)); % sqrt(a_N * p^(N+1))
    
    if iRan > 0 % Only need a_1, a_2, ...
        
        bitHASH = double(randi(2, iRan, D) - 1.5 ) * 2;  % Matrix of R x D    
        temp    = DATA * bitHASH';         

        DATA_SKETCH(:, i) = const .* prod(temp, 2); 
        
    else
        
        DATA_SKETCH(:, i) = const .* ones(N, 1);
        
    end
end

% Scaling
DATA_SKETCH = DATA_SKETCH / sqrt(CS_COL);

