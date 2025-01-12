import Sdk from "casdoor-js-sdk";

export const ServerUrl = "https://flashwords.ydns.eu";

const sdkConfig = {
  serverUrl: "https://flashwords.ydns.eu",
  clientId: "26bb0c442dd7c4bb8dbc",
  organizationName: "built-in",
  appName: "application_vocabularius",
  redirectPath: "/callback"
  // signinPath: "/api/signin",
};

export const CasdoorSDK = new Sdk(sdkConfig);

export const isLoggedIn = () => {
  const token = getToken();
  return token !== null && token.length > 0;
};

export const setToken = (token: string) => {
  localStorage.setItem("accessToken", token);
};

export const getToken = () => {
  return localStorage.getItem("accessToken");
}

export const goToLink = (link: string) => {
  window.location.href = link;
};

export const getUserinfo = () => {
  //@ts-ignore
  return CasdoorSDK.getUserInfo(getToken());
};

export const goToProfilePage = () => {
  //@ts-ignore
  window.location.assign(CasdoorSDK.getMyProfileUrl());
}

export const logout = () => {
  localStorage.removeItem("accessToken");
  let allCookies = document.cookie.split(';');

  // The "expire" attribute of every cookie is
  // Set to "Thu, 01 Jan 1970 00:00:00 GMT"
  for (let i = 0; i < allCookies.length; i++)
    document.cookie = allCookies[i] + "=;expires="
        + new Date(0).toUTCString();

  window.location.reload();
};

export const showMessage = (message: string) => {
  alert(message);
};

