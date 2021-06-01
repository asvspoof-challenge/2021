function b = hp(fs)

% Equiripple Highpass filter designed using the FIRPM function.
% All frequency values are in Hz.

Fstop = 0.125;           % Stopband Frequency
Fpass = 0.25;            % Passband Frequency
Dstop = 0.001;           % Stopband Attenuation
Dpass = 0.057501127785;  % Passband Ripple
dens  = 20;              % Density Factor

% Calculate the order from the parameters using FIRPMORD.
[N, Fo, Ao, W] = firpmord([Fstop, Fpass]/(fs/2), [0 1], [Dstop, Dpass]);

% Calculate the coefficients using the FIRPM function.
b  = firpm(N, Fo, Ao, W, {dens});

end