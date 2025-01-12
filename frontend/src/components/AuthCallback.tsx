import {useEffect} from "react";
import * as Setting from "../Setting";
import {setToken} from "../Setting";
import React from 'react';

export const AuthCallback = () => {
  useEffect(() => {
    Setting.CasdoorSDK.exchangeForAccessToken().then((res) => {
        setToken(res.access_token);
        Setting.goToLink("/");
      }
    )
  }, []);

  return <div>signing...</div>;
}