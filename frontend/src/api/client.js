/**
 * Veritus — API Client Wrapper (src/api/client.js)
 *
 * Handles HTTP requests to the FastAPI backend, provides standardized
 * error handling, and parses response data.
 */

const BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

/**
 * Standardized fetch helper that parses JSON and provides clean error messages.
 */
async function request(endpoint, options = {}) {
  const url = `${BASE_URL}${endpoint}`;
  try {
    const response = await fetch(url, options);

    if (!response.ok) {
      let errorMessage = `Server returned status ${response.status} (${response.statusText})`;
      try {
        const errorData = await response.json();
        if (errorData?.detail) {
          errorMessage = typeof errorData.detail === 'string'
            ? errorData.detail
            : JSON.stringify(errorData.detail);
        }
      } catch {
        // Response wasn't JSON; keep statusText
      }
      throw new Error(errorMessage);
    }

    return await response.json();
  } catch (error) {
    if (error.name === 'TypeError' && error.message.includes('fetch')) {
      throw new Error(
        `Unable to connect to the Veritus backend at ${BASE_URL}. Please ensure the FastAPI server is running.`
      );
    }
    throw error;
  }
}

/**
 * Fetches status of connected LLM and PostgreSQL database.
 * @returns {Promise<{ ollama: boolean, database: boolean }>}
 */
export async function getSystemStatus() {
  return request('/api/system-status');
}

/**
 * Fetches the run history of analyzed datasets.
 * @returns {Promise<{ datasets: Array<object> }>}
 */
export async function getHistory() {
  return request('/api/history');
}

/**
 * Fetches specific historical run findings by dataset ID.
 * @param {number|string} datasetId
 * @returns {Promise<object>}
 */
export async function getHistoryDetail(datasetId) {
  return request(`/api/history/${datasetId}`);
}

/**
 * Checks the progress and status of an active analysis job.
 * @param {string} jobId
 * @returns {Promise<{ job_id: string, status: string, progress: number, message: string, result?: object, error?: string }>}
 */
export async function getAnalysisStatus(jobId) {
  return request(`/api/analyze/${jobId}/status`);
}

/**
 * Initiates analysis on an uploaded CSV file.
 *
 * @param {File} file - CSV File object
 * @param {number} numHypotheses - Initial hypotheses count
 * @param {boolean} enableRefinement - Refinement loop flag
 * @param {string} jobId - Custom tracking ID
 * @param {boolean} runAsync - If true, returns immediately for polling
 * @returns {Promise<object>}
 */
export async function analyzeDataset(
  file,
  numHypotheses = 6,
  enableRefinement = true,
  jobId = null,
  runAsync = false
) {
  const formData = new FormData();
  formData.append('file', file);

  const params = new URLSearchParams({
    num_hypotheses: String(numHypotheses),
    enable_refinement: String(enableRefinement),
    run_async: String(runAsync),
  });

  if (jobId) {
    params.append('job_id', jobId);
  }

  return request(`/api/analyze?${params.toString()}`, {
    method: 'POST',
    body: formData,
  });
}

/**
 * Submits a natural language question or custom hypothesis to the Veritus query engine.
 *
 * @param {string} question - Plain-English research question
 * @param {number|string|null} datasetId - Optional historical dataset ID
 * @param {File|null} file - Optional uploaded CSV File object
 * @returns {Promise<object>}
 */
export async function askDatasetQuestion(question, datasetId = null, file = null) {
  const formData = new FormData();
  formData.append('question', question);
  if (datasetId) {
    formData.append('dataset_id', String(datasetId));
  }
  if (file) {
    formData.append('file', file);
  }

  return request('/api/query', {
    method: 'POST',
    body: formData,
  });
}
