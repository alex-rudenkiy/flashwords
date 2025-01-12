import axios from 'axios';

const API_URL = '';

export const login = async (credentials) => {
  const response = await axios.post(`${API_URL}/login`, credentials, {
  headers: {
  }
});
  return response.data;
};

export const register = async (credentials) => {
  const response = await axios.post(`${API_URL}/register`, credentials, {});
  return response.data;
};