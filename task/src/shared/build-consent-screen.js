/**
 * build-consent-screen.js
 * Builds the informed-consent jsPsych timeline node.
 *
 * Ordered disclosure (mirrors plugin-tutorial-intro-*.js's progressive-reveal
 * pattern): box 0 is active immediately; box 1 stays locked (showing a
 * "· · ·" placeholder, not clickable) until box 0 is revealed. The
 * name/checkbox section stays behind its own "· · ·" locked placeholder until
 * both boxes are revealed. Boxes are stacked vertically (not side by side)
 * to make the top-to-bottom reading/reveal order visually obvious.
 *
 * Two boxes (not three) — a third box repeating Prolific's own
 * timing/compensation listing was removed as redundant. Both remaining boxes
 * are warnings (data loss, session termination), so both carry a plain-text
 * "Warning:" label plus a red background/border to make that unmistakable.
 *
 * "Begin experiment" is NOT given a native `disabled` attribute. Disabled
 * buttons never dispatch a `click` event at all in any browser, which meant
 * a premature click produced zero feedback — several pilot participants
 * reported exactly this confusion, and in at least one case the disabled
 * *look* (opacity/color) apparently didn't render either (most likely a
 * browser/extension override neutralizing color-based disabled styling,
 * which is a known fragile pattern to rely on alone).
 *
 * Instead: the button is styled to *look* disabled via a plain CSS class
 * (decoupled from any native pseudo-class rendering), and a capturing-phase
 * click listener on an ancestor intercepts the click BEFORE it reaches the
 * button — jsPsych attaches its own completion listener directly on the
 * button itself (bubble phase), so a capturing ancestor listener runs first.
 * If requirements aren't met, we stopPropagation() (blocking jsPsych's
 * listener from ever firing) silently — no popup message (an earlier version
 * had one; it shifted the layout when shown, so it was removed). If they are
 * met, we let the event through and jsPsych's normal flow proceeds exactly
 * as before.
 */
import jsPsychHtmlButtonResponse from '@jspsych/plugin-html-button-response';

const N_BOXES = 2;

// Note: jsPsych wraps all trial content in `.jspsych-content { text-align:
// center; }`, so these boxes are center-aligned by inheritance even without
// an explicit override — it's made explicit below for clarity, not because
// leaving it off would produce left-aligned text.
const makeBox = (id, realHTML, textAlign) => `
  <div id="${id}" class="consent-info-box" style="position:relative;">
    <span id="${id}-placeholder"
          style="position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);
                 font-size:1.3rem;font-weight:bold;color:#ccc;white-space:nowrap;">
      · · ·
    </span>
    <span id="${id}-real" style="visibility:hidden;font-size:1.4rem;width:100%;${textAlign ? `text-align:${textAlign};` : ''}">${realHTML}</span>
  </div>`;

/**
 * @param {number} tObsMs          observation response deadline, ms
 * @param {number} maxTimeoutsPerTrial
 * @returns {object} jsPsych timeline node
 */
export function buildConsentScreen(tObsMs, maxTimeoutsPerTrial) {
  const BOX0_REAL = `<strong>Warning:</strong> Do not close, refresh, or navigate away during
            the task — your data will be lost and you will not be paid.`;
  const BOX1_REAL = `<strong>Warning:</strong> You must respond within the ${tObsMs / 1000}-second
            time limit — if you repeatedly time out, the experiment will terminate.`;

  return {
    type: jsPsychHtmlButtonResponse,
    stimulus: `
      <div class="consent-outer">
        <h2 style="text-align:center;margin-bottom:1.5rem;font-size:2.2rem;">Informed Consent</h2>
        <div class="consent-scroll" style="background:#fafafa;border:1.5px solid #d1d5db;border-radius:6px;padding:1rem;margin-bottom:0.75rem;">
          <p style="text-align:center;"><strong>CONSENT TO TAKE PART IN RESEARCH</strong><br>Dartmouth College</p>
          <p style="text-align:center;"><strong>Study title:</strong> Reward learning, choice, and attention<br>
          <strong>Principal Investigator:</strong> Alireza Soltani</p>
          <p><strong>You are being asked to take part in a research study.  Taking part in research is voluntary.</strong></p>
          <div style="text-align:left;">
          <p style="margin-top:0.75rem;"><strong>Study Summary:</strong><br>
          The purpose of this study is to better understand how we learn from reward feedback, make decisions based on that learning, and how these processes are influenced by attention. There are no physical or psychological risks associated with this study beyond the potential tiredness from looking at a computer screen and pressing keys on a keyboard. Monitoring with an infrared camera has no known health risks.</p>
          <p>Your decision whether or not to take part will have no effect on the quality of your academic standing or job status. Please ask questions if there is anything about this study that you do not understand.</p>
          <p style="margin-top:0.75rem;"><strong>What is the purpose of this study?</strong><br>
          The purpose of the study is to explore how people learn from rewards, use what they have learned to make decisions, and how their attention is controlled and affects these processes.</p>
          <p style="margin-top:0.75rem;"><strong>Will you benefit from taking part in this study?</strong><br>
          You will not personally benefit from being in this research study.  We hope to gather information that may help people in the future.</p>
          <p style="margin-top:0.75rem;"><strong>What does this study involve?</strong><br>
          Your participation in this study may last up to 6 hours, which is divided across multiple sessions (over two or more days).</p>
          <p>During the study, you will see one or more visual cues (such as images or shapes) or no cues at all, along with additional information displayed on a computer screen. You will then select between two or more choice options based on this information. On some trials, you will receive feedback, which will tell you whether your choice was correct or incorrect, or whether it was rewarded or not. At certain points, you may also be asked to report what you have learned from the cues and how confident you are about your learning or choices. You will use a keyboard, buttons, or your eye movements to make and report your decisions, etc. The visual cues you see may include photographs, drawings, simple geometric shapes, or patterns such as sinewave gratings.</p>
          <p>We may monitor your eye movements and pupil dilation using an infrared camera placed on the table in front of you. You will be asked to rest your chin on a chinrest to minimize head movement during this process. The infrared camera works like a regular camera but is sensitive to infrared light. Additionally, you may be asked to complete a questionnaire about your learning or performance on the task.</p>
          <p style="margin-top:0.75rem;"><strong>What are the options if you do not want to take part in this study?</strong><br>
          The alternative is not to take part in this study.</p>
          <p style="margin-top:0.75rem;"><strong>If you take part in this study, what activities will be done only for research purposes?</strong><br>
          If you take part in this study, the following activities will be done only for research purposes: monitoring your eye movements, recording your choices and responses on the keyboard.</p>
          <p style="margin-top:0.75rem;"><strong>What are the risks involved with being enrolled in this study?</strong><br>
          This study does not involve any physical or psychological side effects or risks, other than potential tiredness from looking at a computer screen and pressing keys on a keyboard. You are free to stop participating at any time.</p>
          <p>There are no known health risks associated with monitoring eye movements using an infrared camera.</p>
          <p>All data collected during this study will remain confidential, and your name will not appear in any reports or publications resulting from this research.</p>
          <p style="margin-top:0.75rem;"><strong>Will my data be deidentified and used in the future for other purposes?</strong><br>
          Your data (choice and responses on the keyboard and eye movements) might be stripped of identifiers and used for future research.
          Any future research that uses your data will be reviewed by the Committee for the Protection of Human Subjects at Dartmouth College, who will determine if the research requires your permission or may be properly done without further permission from you.</p>
          <p style="margin-top:0.75rem;"><strong>Other important items you should know:</strong></p>
          <ul style="margin:0.25rem 0 0.5rem 1.5rem;padding:0;">
            <li><strong>Leaving the study:</strong>  You may choose to stop taking part in this study at any time. If you decide to stop taking part, it will have no effect on your academic standing, or job status.</li>
            <li><strong>Number of people in this study:</strong>  We expect (500) people to enroll in this study here, and (0) at other study sites.</li>
            <li><strong>Funding:</strong>  National Science Foundation provides funding to Dartmouth College for this research.</li>
          </ul>
          <p style="margin-top:0.75rem;"><strong>How will your privacy be protected?</strong></p>
          <p style="margin-top:0;"><strong>The information collected as data for this study includes:</strong><br>
          The data collected in this study include behavioral measures, such as button-press data, eye movements, and pupil dilation. Your name and age will be collected along with your answers to the questionnaire.  We will also record your gender, handedness, age, and other basic information that will not and cannot be used to identify anything private about you. We may also ask for your average household income to better understand how the incentive provided by compensation from the experiment could affect your performance.</p>
          <p>Data collected for this study will be maintained indefinitely.</p>
          <p>We are careful to protect the identities of the people in this study.  We also keep the information collected for this study secure and confidential.  Behavioral data and responses from computer-based questionnaires will be stored on password-protected computers in the lab and identified only by randomized IDs.</p>
          <p>If research data is shared electronically or physically outside the institution, all identifying information will be removed to ensure anonymity.</p>
          <p style="margin-top:0.75rem;"><strong>Will you be paid to take part in this study?</strong><br>
          Yes or no.  You will be paid for your participation based on the number of hours you participate in the experiment, your performance, and what options you choose during the experiment (overall $10-20/hour). You may also choose to instead get one T-point for every hour of experimental data collection. Because of the nature of online experiments, the overall payment for online version of the experiment will be $5-10/hour.</p>
          <p style="margin-top:0.75rem;"><strong>Whom should you call with questions about this study?</strong><br>
          If you have questions about this study or concerns about a research related problem or injury, you can call the research director for this study: Dr. Alireza Soltani (603) 646-2998 during normal business hours.</p>
          <p>If you have questions, concerns, complaints, or suggestions about human research at Dartmouth, you may call the Office of the Committee for the Protection of Human Subjects at Dartmouth College (603) 646-6482 during normal business hours.</p>
          </div>
        </div>
        <div class="consent-info-boxes">
          ${makeBox('reveal-box-0', BOX0_REAL, 'center')}
          ${makeBox('reveal-box-1', BOX1_REAL, 'center')}
        </div>
        <div class="consent-footer" style="margin-top:0.75rem;position:relative;">
          <!-- consent-fields-real is ALWAYS in flow (so this block's height is
               fixed from the start) — consent-fields-locked is an absolutely
               positioned overlay on top of it. Hiding the overlay reveals what
               was already there underneath, so nothing below it reflows. -->
          <div id="consent-fields-real">
            <!-- PILOT ONLY: name field for within-participant ID (remove for Prolific production) -->
            <div style="display:flex;align-items:center;gap:0.75rem;margin-bottom:0.75rem;justify-content:center;">
              <label for="pilot-name" style="font-size:1.4rem;flex-shrink:0;">Name:</label>
              <input type="text" id="pilot-name" placeholder="Enter your name"
                style="font-size:1.4rem;padding:0.4rem 0.75rem;border:1.5px solid #d1d5db;
                       border-radius:6px;width:220px;">
            </div>
            <label style="display:flex;align-items:center;justify-content:center;gap:0.75rem;cursor:pointer;">
              <input type="checkbox" id="consent-checkbox"
                style="margin-top:3px;width:18px;height:18px;flex-shrink:0;">
              <span style="font-size:1.4rem;">I have read the above information and I agree to take part in this study.</span>
            </label>
          </div>
          <div id="consent-fields-locked" style="
            position:absolute;inset:0;display:flex;align-items:center;justify-content:center;
            background:#f5f5f5;">
            <span style="font-size:1.3rem;color:#ccc;font-weight:bold;">· · ·</span>
          </div>
        </div>
      </div>`,
    choices: ['Begin tutorial'],
    button_html: (c) =>
      `<button id="consent-btn" class="jspsych-btn consent-btn-locked"
        style="font-size:1.6rem;padding:1rem 3.5rem;margin-top:1.5rem;">${c}</button>`,
    data: { screen: 'consent' },
    on_load: () => {
      let revealedCount = 0;
      let fieldsUnlocked = false;
      let _pilotName = '';

      const btn = document.getElementById('consent-btn');

      const setBtnLookReady = (ready) => {
        if (!btn) return;
        btn.classList.toggle('consent-btn-locked', !ready);
      };

      const isReady = () => {
        const checked = document.getElementById('consent-checkbox')?.checked;
        const name    = document.getElementById('pilot-name')?.value.trim() || '';
        return fieldsUnlocked && checked && !!name;
      };

      const refreshBtnLook = () => setBtnLookReady(isReady());

      // Capturing-phase listener on an ancestor — fires BEFORE jsPsych's own
      // bubble-phase click listener on the button itself (jsPsych attaches
      // that during trial() setup, before on_load runs). If requirements
      // aren't met, silently stop it here; otherwise let it through untouched.
      const interceptor = (e) => {
        if (!btn || !e.target || !e.target.closest('#consent-btn')) return;
        if (!isReady()) {
          e.preventDefault();
          e.stopPropagation();
        }
      };
      document.addEventListener('click', interceptor, { capture: true });

      const revealBox = (i) => {
        document.getElementById('reveal-box-' + i + '-placeholder').style.display      = 'none';
        document.getElementById('reveal-box-' + i + '-real').style.visibility = 'visible';
        const box = document.getElementById('reveal-box-' + i);
        box.style.cursor = 'default';
        revealedCount++;

        if (revealedCount < N_BOXES) {
          activateBox(revealedCount);
        } else {
          // All boxes done — unlock the name/checkbox section (already in
          // flow; just hide the overlay, no layout shift).
          fieldsUnlocked = true;
          document.getElementById('consent-fields-locked').style.display = 'none';
          const nameInput = document.getElementById('pilot-name');
          const checkbox  = document.getElementById('consent-checkbox');
          nameInput.addEventListener('input', refreshBtnLook);
          checkbox.addEventListener('change', refreshBtnLook);
        }
        refreshBtnLook();
      };

      const activateBox = (i) => {
        const box = document.getElementById('reveal-box-' + i);
        const ph  = document.getElementById('reveal-box-' + i + '-placeholder');
        box.style.cursor = 'pointer';
        if (ph) { ph.style.color = '#555'; ph.textContent = 'Click to reveal'; }
        box.addEventListener('click', () => revealBox(i), { once: true });
      };

      // Box 0 starts active; box 1 starts locked ("· · ·") until box 0 is
      // revealed — mirrors plugin-tutorial-intro-*.js.
      activateBox(0);
      refreshBtnLook();

      // Exposed for on_finish to read the captured name and for cleanup.
      window._pilotNameCapture = null;
      const nameCaptureUpdater = () => {
        _pilotName = document.getElementById('pilot-name')?.value.trim() || '';
        window._pilotNameCapture = _pilotName;
      };
      document.addEventListener('input', nameCaptureUpdater);

      window._consentCleanup = () => {
        document.removeEventListener('click', interceptor, { capture: true });
        document.removeEventListener('input', nameCaptureUpdater);
      };
    },
    on_finish: (data) => {
      data.consent_given = true;
      // PILOT ONLY: save name as prolific_pid substitute (remove for Prolific production)
      if (window._pilotNameCapture) {
        data.pilot_name = window._pilotNameCapture;
        window._pilotNameCapture = null;
      }
      if (window._consentCleanup) {
        window._consentCleanup();
        window._consentCleanup = null;
      }
    },
  };
}
