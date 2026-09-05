import './shared/jatos-shim.js';
import { config } from './continuous/config.js';
import { buildAndRun } from './shared/timeline-builder.js';

buildAndRun(config);
